"""
Distil-Whisper STT Service — Flash Attention 2 + batched chunked inference.

Two models available (set via MODEL env var or request param):
- distil-whisper/distil-large-v3: 756M params, 6x faster, within 1% WER
- openai/whisper-large-v3: 1.5B params, highest accuracy

Performance (A10G, 150 min audio):
- distil-large-v3 + FA2 + batch=24: ~75s (120x realtime)
- whisper-large-v3 + FA2 + batch=24: ~95s (95x realtime)
- Without FA2 + batch=16: ~5-10min

Deploy: modal deploy modal/distil_whisper_service.py
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("modal-inference-distil-whisper")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"

DISTIL_MODEL = "distil-whisper/distil-large-v3"
FULL_MODEL = "openai/whisper-large-v3"
DEFAULT_MODEL = os.environ.get("STT_MODEL", DISTIL_MODEL)
DEFAULT_BATCH_SIZE = 24

# --- Download models at image build time ------------------------------------

def download_models():
    """Download both Whisper models during image build."""
    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    for model_id in [DISTIL_MODEL, FULL_MODEL]:
        print(f"Downloading {model_id}...")
        AutoProcessor.from_pretrained(model_id)
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    print("All STT models downloaded.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "transformers>=4.45",
        "accelerate",
        "flash-attn>=2.6.0",
    )
    .run_function(download_models)
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=300,
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=10,
)
class DistilWhisperService:
    """Speech-to-text via Whisper with Flash Attention 2 + batched inference."""

    @modal.enter()
    def load_model(self):
        """Load default model with flash attention and build pipeline."""
        self.pipes = {}
        self._load(DEFAULT_MODEL)

    def _load(self, model_id: str):
        """Load a whisper model into the pipeline cache."""
        if model_id in self.pipes:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        os.environ["HF_HOME"] = HF_CACHE

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipes[model_id] = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.device = device
        print(f"Loaded {model_id} with flash_attention_2 on {device}.")

    def get_pipe(self, model_id: str | None = None):
        mid = model_id or DEFAULT_MODEL
        if mid not in self.pipes:
            self._load(mid)
        return self.pipes[mid]

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with transcription endpoint."""
        import time
        import traceback
        import base64
        import soundfile as sf
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "default_model": DEFAULT_MODEL,
                "loaded_models": list(svc.pipes.keys()),
                "available_models": [DISTIL_MODEL, FULL_MODEL],
                "gpu": "A10G",
                "flash_attention": True,
                "default_batch_size": DEFAULT_BATCH_SIZE,
                "capabilities": ["transcribe", "timestamps", "multilingual", "long_form"],
            }

        @api.post("/transcribe")
        async def transcribe(request: Request):
            """Transcribe audio to text.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - timestamps (str, optional): "word" for word-level, "chunk" for chunks (default "chunk")
              - language (str, optional): language hint (e.g. "en", "fr")
              - chunk_length_s (int, optional): chunk length for long-form (default 30)
              - batch_size (int, optional): batch size for chunked inference (default 16)

            Returns:
              {
                "text": "full transcription...",
                "chunks": [{"text": "...", "timestamp": [0.0, 2.5]}],
                "language": "en",
                "duration_seconds": 120.5,
                "processing_time": 2.1
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            timestamps = body.get("timestamps", "chunk")
            language = body.get("language")
            chunk_length_s = body.get("chunk_length_s", 30)
            batch_size = body.get("batch_size", DEFAULT_BATCH_SIZE)
            model_id = body.get("model")

            if not audio_url and not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_url or audio_file (base64)"},
                )

            temp_path = None
            try:
                audio_bytes = None
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)

                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=16000,
                    mono=True,
                )

                # Build generate kwargs
                generate_kwargs = {}
                if language:
                    generate_kwargs["language"] = language

                # Determine return_timestamps value
                return_timestamps = timestamps if timestamps == "word" else True

                pipe = svc.get_pipe(model_id)
                output = pipe(
                    temp_path,
                    chunk_length_s=chunk_length_s,
                    batch_size=batch_size,
                    return_timestamps=return_timestamps,
                    generate_kwargs=generate_kwargs,
                )

                result = {
                    "text": output.get("text", "").strip(),
                }

                # Include chunks with timestamps
                if "chunks" in output:
                    result["chunks"] = [
                        {
                            "text": chunk["text"].strip(),
                            "timestamp": list(chunk["timestamp"]) if chunk.get("timestamp") else None,
                        }
                        for chunk in output["chunks"]
                    ]

                # Compute duration from audio
                info = sf.info(temp_path)
                result["duration_seconds"] = round(info.duration, 3)

                elapsed = time.time() - start_time
                result["processing_time"] = round(elapsed, 3)

                rtf = elapsed / info.duration if info.duration > 0 else 0
                used_model = model_id or DEFAULT_MODEL
                result["model"] = used_model
                print(
                    f"stt [{used_model}]: {info.duration:.1f}s audio "
                    f"→ {elapsed:.3f}s (RTF={rtf:.4f}, batch={batch_size})"
                )

                return result

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        return api


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Distil-Whisper STT Service ready")
    print("Deploy with: modal deploy modal/distil_whisper_service.py")
