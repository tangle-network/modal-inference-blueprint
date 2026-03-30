"""
NeMo Sortformer Diarization Service - End-to-end Transformer speaker diarization.

Capabilities:
- End-to-end speaker diarization (no pipeline, single Transformer)
- Streaming-capable (Sortformer v2.1)
- Up to 4 speakers per segment
- NVIDIA NeMo toolkit, Apache 2.0 license

Deploy: modal deploy modal/nemo_diarization_service.py

Requirements:
- A10G GPU (24GB VRAM)
- NeMo toolkit with ASR extras
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("nemo-diarization")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"

DIAR_MODEL_NAME = "nvidia/diar_sortformer_4spk-v1"

# --- Download model at image build time --------------------------------------

def download_nemo_models():
    """Download Sortformer diarization model during image build."""
    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    from nemo.collections.asr.models import SortformerEncLabelModel

    print(f"Downloading {DIAR_MODEL_NAME}...")
    SortformerEncLabelModel.from_pretrained(DIAR_MODEL_NAME)
    print("Sortformer download complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.10"))
    .apt_install("ffmpeg", "libsndfile1", "git", "build-essential")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "Cython",
        "packaging",
        "editdistance",
        "jiwer",
        "omegaconf",
        "hydra-core",
        "pytorch-lightning",
        "transformers>=4.36.0",
        "huggingface_hub",
        "sentencepiece",
        "webdataset",
        "braceexpand",
        "lhotse>=1.22",
    )
    .run_commands(
        "pip install nemo_toolkit[asr]"
    )
    .run_function(download_nemo_models)
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=1800,
    container_idle_timeout=300,
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=5,
)
class NemoDiarizationService:
    """Speaker diarization via NVIDIA NeMo Sortformer."""

    @modal.enter()
    def load_model(self):
        """Load Sortformer model on container start."""
        import torch

        os.environ["HF_HOME"] = HF_CACHE
        os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

        from nemo.collections.asr.models import SortformerEncLabelModel

        self.model = SortformerEncLabelModel.from_pretrained(DIAR_MODEL_NAME)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        print(f"Sortformer ({DIAR_MODEL_NAME}) loaded on GPU.")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with diarization endpoint."""
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
                "model": "sortformer-4spk-v1",
                "framework": "nvidia-nemo",
                "gpu": "A10G",
                "capabilities": [
                    "speaker_diarization",
                    "end_to_end_transformer",
                    "up_to_4_speakers",
                ],
                "max_speakers": 4,
            }

        @api.post("/diarize")
        async def diarize(request: Request):
            """Diarize audio into speaker-labeled segments.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - max_speakers (int, optional): hint for max speakers (1-4, default 4)

            Returns:
              {
                "segments": [
                  {"speaker": "speaker_0", "start": 0.5, "end": 3.2}
                ],
                "speakers": ["speaker_0", "speaker_1"],
                "num_speakers": 2,
                "duration_seconds": 120.5,
                "processing_time": 2.3
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"}
                )

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")

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

                # Run diarization
                predicted = svc.model.diarize(
                    audio=[temp_path],
                    batch_size=1,
                )

                # Parse Sortformer output into structured segments
                # Output is a list (per file) of segment objects
                segments = []
                speakers_set = set()

                if predicted and len(predicted) > 0:
                    file_segments = predicted[0]
                    for seg in file_segments:
                        # Sortformer returns segments as strings or objects
                        # depending on the version. Handle both.
                        if isinstance(seg, str):
                            # Parse RTTM-like format:
                            # "speaker_0 0.500 3.200"
                            parts = seg.strip().split()
                            if len(parts) >= 3:
                                speaker = parts[0]
                                start_t = float(parts[1])
                                end_t = float(parts[2])
                            else:
                                continue
                        elif hasattr(seg, "speaker"):
                            speaker = seg.speaker if hasattr(seg, "speaker") else f"speaker_{seg.label}"
                            start_t = seg.start
                            end_t = seg.end
                        elif isinstance(seg, (list, tuple)):
                            start_t, end_t = float(seg[0]), float(seg[1])
                            speaker = str(seg[2]) if len(seg) > 2 else "speaker_0"
                        else:
                            # Try dict-like access
                            start_t = float(seg.get("start", seg.get("onset", 0)))
                            end_t = float(seg.get("end", seg.get("offset", 0)))
                            speaker = seg.get("speaker", seg.get("label", "speaker_0"))

                        segments.append({
                            "speaker": speaker,
                            "start": round(start_t, 3),
                            "end": round(end_t, 3),
                        })
                        speakers_set.add(speaker)

                info = sf.info(temp_path)
                elapsed = time.time() - start_time

                result = {
                    "segments": segments,
                    "speakers": sorted(speakers_set),
                    "num_speakers": len(speakers_set),
                    "duration_seconds": round(info.duration, 3),
                    "processing_time": round(elapsed, 3),
                }

                print(
                    f"nemo-sortformer: diarized {info.duration:.1f}s audio -> "
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

        return api


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("NeMo Sortformer Diarization Service ready")
    print("Deploy with: modal deploy modal/nemo_diarization_service.py")
