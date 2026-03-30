"""
NVIDIA Parakeet-TDT-0.6B-V3 STT Service - Multilingual speech-to-text.

Capabilities:
- #1 on HuggingFace Open ASR Leaderboard
- 600M params, 25 European languages, auto language detection
- Word/segment/char timestamps via TDT (Token-and-Duration Transducer)
- 3386x real-time factor on GPU
- CC-BY-4.0 license

Deploy: modal deploy modal/parakeet_service.py

Requirements:
- NVIDIA NeMo toolkit (nemo_toolkit[asr])
- A10G GPU (24GB VRAM, model is ~1.2GB)
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("parakeet")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"
NEMO_CACHE = f"{MODEL_CACHE_DIR}/nemo"

# --- Download model at image build time -------------------------------------

def download_parakeet():
    """Download Parakeet-TDT model during image build."""
    import os
    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs(NEMO_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TORCH_HOME"] = NEMO_CACHE

    import nemo.collections.asr as nemo_asr
    print("Downloading nvidia/parakeet-tdt-0.6b-v3...")
    nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
    )
    print("Parakeet-TDT download complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
    )
    .pip_install(
        "Cython",
        "packaging",
    )
    .pip_install(
        "nemo_toolkit[asr]>=2.0.0",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
    )
    .run_function(download_parakeet)
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
class ParakeetService:
    """Speech-to-text via NVIDIA Parakeet-TDT-0.6B-V3."""

    @modal.enter()
    def load_model(self):
        """Load Parakeet model on container start."""
        os.environ["HF_HOME"] = HF_CACHE
        os.environ["TORCH_HOME"] = NEMO_CACHE

        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
        )
        self.model.eval()
        print("Parakeet-TDT-0.6B-V3 loaded on GPU.")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with transcription endpoint."""
        import time
        import traceback
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "parakeet-tdt-0.6b-v3",
                "gpu": "A10G",
                "capabilities": ["transcribe", "timestamps", "multilingual"],
                "languages": 25,
            }

        @api.post("/transcribe")
        async def transcribe(request: Request):
            """Transcribe audio to text with optional timestamps.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - timestamps (bool, optional): return word/segment timestamps (default true)
              - language (str, optional): hint language code (auto-detected if omitted)

            Returns:
              {
                "text": "full transcription...",
                "language": "en",
                "segments": [{"text": "...", "start": 0.0, "end": 1.5}],
                "words": [{"word": "hello", "start": 0.0, "end": 0.3}],
                "duration_seconds": 120.5,
                "processing_time": 0.8
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            include_timestamps = body.get("timestamps", True)

            if not audio_url and not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_url or audio_file (base64)"},
                )

            temp_path = None
            try:
                audio_bytes = None
                if audio_b64:
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)

                # Parakeet expects 16kHz mono WAV
                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=16000,
                    mono=True,
                )

                # Transcribe
                output = svc.model.transcribe(
                    [temp_path],
                    timestamps=include_timestamps,
                )

                hypothesis = output[0]
                text = hypothesis.text

                result = {
                    "text": text,
                }

                # Extract timestamps if available
                if include_timestamps and hasattr(hypothesis, "timestamp") and hypothesis.timestamp:
                    ts = hypothesis.timestamp

                    # Segment-level timestamps
                    if "segment" in ts:
                        result["segments"] = [
                            {
                                "text": seg.get("segment", seg.get("text", "")),
                                "start": round(seg["start"], 3),
                                "end": round(seg["end"], 3),
                            }
                            for seg in ts["segment"]
                        ]

                    # Word-level timestamps
                    if "word" in ts:
                        result["words"] = [
                            {
                                "word": w.get("word", w.get("char", "")),
                                "start": round(w["start"], 3),
                                "end": round(w["end"], 3),
                            }
                            for w in ts["word"]
                        ]

                # Detect language from hypothesis if available
                if hasattr(hypothesis, "lang") and hypothesis.lang:
                    result["language"] = hypothesis.lang

                # Compute duration from audio file
                import soundfile as sf
                info = sf.info(temp_path)
                result["duration_seconds"] = round(info.duration, 3)

                elapsed = time.time() - start_time
                result["processing_time"] = round(elapsed, 3)

                rtf = elapsed / info.duration if info.duration > 0 else 0
                print(
                    f"parakeet: transcribed {info.duration:.1f}s audio "
                    f"in {elapsed:.3f}s (RTF={rtf:.4f})"
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
    print("Parakeet-TDT STT Service ready")
    print("Deploy with: modal deploy modal/parakeet_service.py")
