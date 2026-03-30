"""
RVC v2 Voice Conversion Service - Real-time voice conversion with retrieval.

Capabilities:
- Real-time voice conversion (speaker → target voice)
- Retrieval-based approach with FAISS index for quality
- HuBERT feature extraction + pitch estimation (RMVPE)
- 50K+ stars, most popular open-source voice conversion
- MIT license

Deploy: modal deploy modal/rvc_service.py

Requirements:
- A10G GPU (24GB VRAM)
- Trained RVC v2 .pth model files uploaded via /register_voice
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("rvc")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
VOICE_DIR = "/voice-cache"
VOICE_MODELS_DIR = f"{VOICE_DIR}/models"
VOICE_INDEX_DIR = f"{VOICE_DIR}/indexes"

rvc_volume = modal.Volume.from_name("phony-rvc-voices", create_if_missing=True)

# --- Download pretrained assets at image build time --------------------------

def download_rvc_assets():
    """Download HuBERT base model and RVC pretrained weights."""
    import os
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    from huggingface_hub import hf_hub_download

    # HuBERT base model for feature extraction
    print("Downloading HuBERT base model...")
    hf_hub_download(
        repo_id="lj1995/VoiceConversionWebUI",
        filename="hubert_base.pt",
        local_dir=MODEL_CACHE_DIR,
    )

    # RMVPE pitch estimator
    print("Downloading RMVPE pitch estimator...")
    hf_hub_download(
        repo_id="lj1995/VoiceConversionWebUI",
        filename="rmvpe.pt",
        local_dir=MODEL_CACHE_DIR,
    )

    print("RVC asset downloads complete.")


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
        "faiss-cpu",
        "scipy",
        "librosa",
        "praat-parselmouth",
        "pyworld",
        "torchcrepe",
        "huggingface_hub",
    )
    .run_commands(
        "pip install git+https://github.com/daswer123/rvc-python.git"
    )
    .run_function(download_rvc_assets)
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=300,
    volumes={
        MODEL_CACHE_DIR: model_cache,
        VOICE_DIR: rvc_volume,
    },
    allow_concurrent_inputs=5,
)
class RVCService:
    """Real-time voice conversion via RVC v2."""

    @modal.enter()
    def load_model(self):
        """Load RVC inference engine on container start."""
        os.makedirs(VOICE_MODELS_DIR, exist_ok=True)
        os.makedirs(VOICE_INDEX_DIR, exist_ok=True)

        from rvc_python.infer import RVCInference

        self.rvc = RVCInference(device="cuda:0")
        self._loaded_voice = None
        print("RVC v2 inference engine loaded on GPU.")

    def _ensure_voice_loaded(self, voice_id: str):
        """Load an RVC model for the given voice if not already loaded."""
        if self._loaded_voice == voice_id:
            return

        model_path = f"{VOICE_MODELS_DIR}/{voice_id}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model '{voice_id}' not found")

        index_path = f"{VOICE_INDEX_DIR}/{voice_id}.index"
        index = index_path if os.path.exists(index_path) else ""

        self.rvc.load_model(model_path, index_path=index)
        self._loaded_voice = voice_id

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with voice conversion endpoints."""
        import time
        import traceback
        import base64
        import io
        import tempfile
        import soundfile as sf
        from fastapi import FastAPI, Request, UploadFile, File, Form
        from fastapi.responses import Response, JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            voices = []
            if os.path.exists(VOICE_MODELS_DIR):
                voices = [
                    f[:-4] for f in os.listdir(VOICE_MODELS_DIR)
                    if f.endswith(".pth")
                ]
            return {
                "status": "ok",
                "model": "rvc-v2",
                "gpu": "A10G",
                "capabilities": [
                    "voice_conversion", "pitch_shift",
                    "retrieval_index", "rmvpe",
                ],
                "registered_voices": len(voices),
            }

        @api.post("/convert")
        async def convert(request: Request):
            """Convert audio to a target voice.

            Accepts JSON:
              - audio_url (str): URL to source audio
              - audio_file (str, base64): source audio bytes (base64-encoded)
              - target_voice_id (str): registered RVC model ID
              - pitch_shift (int, optional): semitones to shift pitch (default 0)
              - f0_method (str, optional): pitch estimation method (default "rmvpe")
              - index_rate (float, optional): retrieval index influence 0-1 (default 0.75)
              - filter_radius (int, optional): median filter radius for pitch (default 3)
              - rms_mix_rate (float, optional): volume envelope mix 0-1 (default 0.25)
              - protect (float, optional): consonant protection 0-0.5 (default 0.33)

            Returns: WAV audio with converted voice.
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            target_voice_id = body.get("target_voice_id")

            if not audio_url and not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_url or audio_file (base64)"},
                )
            if not target_voice_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "target_voice_id is required"},
                )

            pitch_shift = body.get("pitch_shift", 0)
            f0_method = body.get("f0_method", "rmvpe")
            index_rate = body.get("index_rate", 0.75)
            filter_radius = body.get("filter_radius", 3)
            rms_mix_rate = body.get("rms_mix_rate", 0.25)
            protect = body.get("protect", 0.33)

            input_path = None
            output_path = None
            try:
                audio_bytes = None
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)

                input_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=16000,
                    mono=True,
                )

                svc._ensure_voice_loaded(target_voice_id)

                # Configure conversion params
                svc.rvc.set_params(
                    f0method=f0_method,
                    f0up_key=pitch_shift,
                    index_rate=index_rate,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                )

                output_path = tempfile.mktemp(suffix=".wav")
                svc.rvc.infer_file(input_path, output_path)

                with open(output_path, "rb") as f:
                    wav_data = f.read()

                info = sf.info(output_path)
                elapsed = time.time() - start_time

                print(
                    f"rvc: converted {info.duration:.1f}s audio "
                    f"with voice '{target_voice_id}' in {elapsed:.3f}s "
                    f"(pitch={pitch_shift}, f0={f0_method})"
                )

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Processing-Time": f"{elapsed:.3f}",
                        "X-Voice-Id": target_voice_id,
                        "X-Audio-Duration": f"{info.duration:.3f}",
                    },
                )

            except FileNotFoundError as e:
                return JSONResponse(status_code=404, content={"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if input_path and os.path.exists(input_path):
                    os.unlink(input_path)
                if output_path and os.path.exists(output_path):
                    os.unlink(output_path)

        @api.post("/train")
        async def train(request: Request):
            """Train a new RVC voice model from audio samples.

            Not yet implemented — training requires extended GPU time
            and dataset preparation. Use pre-trained models via /register_voice.
            """
            return JSONResponse(
                status_code=501,
                content={
                    "error": "Training not yet implemented. "
                    "Upload pre-trained .pth models via /register_voice."
                },
            )

        @api.post("/register_voice")
        async def register_voice(
            voice_id: str = Form(...),
            model_file: UploadFile = File(...),
            index_file: UploadFile = File(None),
        ):
            """Register a trained RVC model for voice conversion.

            Form fields:
              - voice_id (str): unique identifier
              - model_file (file): .pth RVC model file (required)
              - index_file (file): .index FAISS retrieval index (optional)
            """
            try:
                model_path = f"{VOICE_MODELS_DIR}/{voice_id}.pth"
                model_data = await model_file.read()
                with open(model_path, "wb") as f:
                    f.write(model_data)

                has_index = False
                if index_file:
                    index_path = f"{VOICE_INDEX_DIR}/{voice_id}.index"
                    index_data = await index_file.read()
                    with open(index_path, "wb") as f:
                        f.write(index_data)
                    has_index = True

                rvc_volume.commit()

                # Reset loaded voice so next convert picks up the new model
                svc._loaded_voice = None

                return {
                    "status": "success",
                    "voice_id": voice_id,
                    "model_size_bytes": len(model_data),
                    "has_index": has_index,
                }

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.get("/voices")
        def list_voices():
            """List registered RVC voice models."""
            voices = []
            if os.path.exists(VOICE_MODELS_DIR):
                for f in os.listdir(VOICE_MODELS_DIR):
                    if f.endswith(".pth"):
                        vid = f[:-4]
                        has_index = os.path.exists(
                            f"{VOICE_INDEX_DIR}/{vid}.index"
                        )
                        model_size = os.path.getsize(
                            f"{VOICE_MODELS_DIR}/{f}"
                        )
                        voices.append({
                            "id": vid,
                            "has_index": has_index,
                            "model_size_bytes": model_size,
                        })
            return {"voices": voices}

        return api


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("RVC v2 Voice Conversion Service ready")
    print("Deploy with: modal deploy modal/rvc_service.py")
