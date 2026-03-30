"""
Pyannote 3.1 Diarization Service - Speaker diarization and embedding extraction.

Capabilities:
- Speaker diarization (pyannote/speaker-diarization-3.1)
- Speaker embedding extraction (pyannote/embedding)
- Pure PyTorch (no onnxruntime dependency)
- Apache 2.0 license (pyannote.audio), gated models require HF acceptance

Deploy: modal deploy modal/pyannote_service.py

Note: pyannote models are gated on HuggingFace. You must:
  1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
  2. Accept terms at https://huggingface.co/pyannote/segmentation-3.0
  3. Accept terms at https://huggingface.co/pyannote/embedding
  4. Store HF_TOKEN in Modal secret named "huggingface"
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("pyannote")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"

# --- Download models at image build time ------------------------------------

def download_pyannote_models():
    """Download pyannote pipeline + embedding model during image build."""
    import os
    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: No HF_TOKEN — pyannote models are gated, download will fail")
        return

    from pyannote.audio import Pipeline, Inference
    print("Downloading pyannote/speaker-diarization-3.1...")
    Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    print("Downloading pyannote/embedding...")
    Inference("pyannote/embedding", use_auth_token=hf_token, window="whole")
    print("Pyannote model downloads complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
    )
    .pip_install(
        "pyannote.audio>=3.1.1,<4",
        "speechbrain>=1.0.0",
        "transformers>=4.36.0",
        "scipy",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
    )
    .run_function(
        download_pyannote_models,
        secrets=[modal.Secret.from_name("huggingface")],
    )
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=1800,
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=5,
)
class PyannoteService:
    """Speaker diarization and embedding extraction via pyannote.audio 3.1."""

    @modal.enter()
    def load_models(self):
        """Load diarization pipeline and embedding model on container start."""
        import torch

        os.environ["HF_HOME"] = HF_CACHE
        os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
        hf_token = os.environ.get("HF_TOKEN")

        from pyannote.audio import Pipeline, Inference

        # Diarization pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        self.pipeline.to(torch.device("cuda"))

        # Speaker embedding model
        self.embedding_model = Inference(
            "pyannote/embedding",
            use_auth_token=hf_token,
            window="whole",
        )
        self.embedding_model.to(torch.device("cuda"))

        print("Pyannote models loaded on GPU.")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with diarization and embedding endpoints."""
        import time
        import traceback
        import torchaudio
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "pyannote-diarization-3.1",
                "gpu": "A10G",
                "capabilities": ["diarize", "identify"],
            }

        @api.post("/diarize")
        async def diarize(request: Request):
            """Diarize audio into speaker-labeled segments.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - min_speakers (int, optional): minimum expected speakers
              - max_speakers (int, optional): maximum expected speakers
              - return_embeddings (bool, optional): include per-speaker embeddings

            Returns:
              {
                "segments": [{"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2}],
                "speakers": ["SPEAKER_00", "SPEAKER_01"],
                "duration_seconds": 120.5,
                "processing_time": 2.3
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            min_speakers = body.get("min_speakers")
            max_speakers = body.get("max_speakers")
            return_embeddings = body.get("return_embeddings", False)

            if not audio_url and not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_url or audio_file (base64)"},
                )

            temp_path = None
            try:
                # Decode base64 if provided
                audio_bytes = None
                if audio_b64:
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)

                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=16000,
                    mono=True,
                )

                waveform, sr = torchaudio.load(temp_path)

                # Build pipeline kwargs
                pipeline_kwargs = {}
                if min_speakers is not None:
                    pipeline_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    pipeline_kwargs["max_speakers"] = max_speakers

                diarization = svc.pipeline(
                    {"waveform": waveform, "sample_rate": sr},
                    **pipeline_kwargs,
                )

                # Extract segments
                segments = []
                speakers_set = set()
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append({
                        "speaker": speaker,
                        "start": round(turn.start, 3),
                        "end": round(turn.end, 3),
                    })
                    speakers_set.add(speaker)

                duration = waveform.shape[1] / sr
                elapsed = time.time() - start_time

                result = {
                    "segments": segments,
                    "speakers": sorted(speakers_set),
                    "num_speakers": len(speakers_set),
                    "duration_seconds": round(duration, 3),
                    "processing_time": round(elapsed, 3),
                }

                # Per-speaker embeddings via cropping
                if return_embeddings and segments:
                    speaker_embeddings = {}
                    for speaker in sorted(speakers_set):
                        # Find longest segment for this speaker
                        speaker_segs = [
                            s for s in segments if s["speaker"] == speaker
                        ]
                        longest = max(speaker_segs, key=lambda s: s["end"] - s["start"])

                        from pyannote.core import Segment
                        crop = Segment(longest["start"], longest["end"])
                        emb = svc.embedding_model.crop(
                            {"waveform": waveform, "sample_rate": sr},
                            crop,
                        )
                        speaker_embeddings[speaker] = emb.data.squeeze().tolist()

                    result["embeddings"] = speaker_embeddings

                print(
                    f"pyannote: diarized {duration:.1f}s audio -> "
                    f"{len(speakers_set)} speakers, {len(segments)} segments "
                    f"in {elapsed:.3f}s"
                )

                return result

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        @api.post("/identify")
        async def identify(request: Request):
            """Extract speaker embeddings from audio.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - segments (list, optional): specific time ranges to extract
                  [{"start": 0.0, "end": 5.0, "label": "speaker_a"}]
                If omitted, extracts a single embedding for the entire file.

            Returns:
              {
                "embeddings": [
                  {"label": "whole" | "speaker_a", "embedding": [0.1, ...], "dimension": 512}
                ]
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            segments_spec = body.get("segments")

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

                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=16000,
                    mono=True,
                )

                waveform, sr = torchaudio.load(temp_path)
                audio_input = {"waveform": waveform, "sample_rate": sr}

                embeddings = []

                if segments_spec:
                    from pyannote.core import Segment
                    for seg in segments_spec:
                        crop = Segment(seg["start"], seg["end"])
                        emb = svc.embedding_model.crop(audio_input, crop)
                        data = emb.data.squeeze().tolist()
                        embeddings.append({
                            "label": seg.get("label", f"{seg['start']}-{seg['end']}"),
                            "start": seg["start"],
                            "end": seg["end"],
                            "embedding": data,
                            "dimension": len(data),
                        })
                else:
                    emb = svc.embedding_model(audio_input)
                    data = emb.data.squeeze().tolist()
                    embeddings.append({
                        "label": "whole",
                        "embedding": data,
                        "dimension": len(data),
                    })

                elapsed = time.time() - start_time
                print(
                    f"pyannote: extracted {len(embeddings)} embedding(s) "
                    f"in {elapsed:.3f}s"
                )

                return {
                    "embeddings": embeddings,
                    "processing_time": round(elapsed, 3),
                }

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
    print("Pyannote Diarization Service ready")
    print("Deploy with: modal deploy modal/pyannote_service.py")
