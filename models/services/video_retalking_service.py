"""
Video-Retalking Lip-Sync Re-Dubbing Service — Video + Audio → Re-Dubbed Video.

High-quality lip-sync re-dubbing on existing video. Takes a source video
and target audio, generates a re-dubbed video with matched lip movements.
Uses the OpenTalker/video-retalking model on A10G.

Endpoints:
  POST /generate  — video_url + audio_url → re-dubbed video bytes (MP4)
  GET  /health    — service status

Run:   modal run infra/modal-gpu/video_retalking_service.py
Deploy: modal deploy infra/modal-gpu/video_retalking_service.py
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile, base_service_layer

app = create_modal_app("video-retalking")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
RETALKING_DIR = f"{MODEL_CACHE_DIR}/video-retalking"
RETALKING_REPO_URL = "https://github.com/OpenTalker/video-retalking.git"


def download_models():
    """Clone video-retalking repo and download pretrained checkpoints at image build time."""
    import subprocess

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    # Clone the repo
    if not os.path.exists(RETALKING_DIR):
        print(f"Cloning {RETALKING_REPO_URL}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", RETALKING_REPO_URL, RETALKING_DIR],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.decode()[:500]}")
        print("video-retalking repo cloned.")

    # Download pretrained checkpoints via the repo's download script or manual fetch
    checkpoints_dir = f"{RETALKING_DIR}/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # The model requires several pretrained weights:
    # - face detection (S3FD)
    # - face parsing (BiseNet)
    # - face enhancement (GFPGAN/RestoreFormer)
    # - lip-sync (LipSync Expert / SyncNet)
    # - DNet, ENet, expression extractor
    from huggingface_hub import hf_hub_download

    # video-retalking checkpoints are hosted on HuggingFace at vinthony/video-retalking
    HF_REPO = "vinthony/video-retalking"
    checkpoint_files = [
        "30_net_gen.pth",
        "BFM.zip",
        "DNet.pt",
        "ENet.pth",
        "expression.mat",
        "face3d_pretrain_epoch_20.pth",
        "GFPGANv1.3.pth",
        "GPEN-BFR-512.pth",
        "LNet.pth",
        "ParseNet-latest.pth",
        "RetinaFace-R50.pth",
        "shape_predictor_68_face_landmarks.dat",
    ]

    for fname in checkpoint_files:
        dest = f"{checkpoints_dir}/{fname}"
        if not os.path.exists(dest):
            print(f"Downloading checkpoint {fname}...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                local_dir=checkpoints_dir,
            )
            print(f"  -> {downloaded}")

    print("video-retalking checkpoints ready.")


image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "opencv-python-headless",
        "Pillow",
        "huggingface_hub",
        "face-alignment",
        "basicsr>=1.4.2",
        "imageio[ffmpeg]",
        "dlib",
        "kornia",
        "ninja",
        "einops",
        "gfpgan",
        "tqdm",
        "scipy",
    )
    .run_function(download_models)
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=180,
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=2,
)
class VideoRetalkingService:
    """Lip-sync re-dubbing via video-retalking — replaces lip and face movements to match new audio."""

    @modal.enter()
    def load_model(self):
        """Load video-retalking models on container start."""
        import sys
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Add repo to path so its modules are importable
        if RETALKING_DIR not in sys.path:
            sys.path.insert(0, RETALKING_DIR)

        self._model_loaded = True
        print(f"video-retalking service ready on {self.device}")

    @modal.asgi_app()
    def web_app(self):
        import time
        import tempfile
        import traceback
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "video-retalking",
                "gpu": "A10G",
                "features": ["lipsync", "redubbing", "face_enhancement"],
                "models": [{
                    "name": "video-retalking",
                    "slug": "video-retalking",
                    "taskType": "video-lipsync",
                    "status": "ready",
                }],
            }

        @api.post("/generate")
        async def generate(request: Request):
            """Re-dub a video with new audio, matching lip movements.

            Accepts JSON:
              - video_url (str): URL to source video
              - audio_url (str): URL to target audio for re-dubbing

            Returns: MP4 video bytes with re-dubbed output
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            video_url = body.get("video_url")
            audio_url = body.get("audio_url")

            if not video_url:
                return JSONResponse(status_code=400, content={"error": "video_url is required"})
            if not audio_url:
                return JSONResponse(status_code=400, content={"error": "audio_url is required"})

            audio_path = None
            video_path = None
            output_path = None
            try:
                # Download inputs
                audio_path = download_audio_to_tempfile(
                    audio_url=audio_url, target_sr=16000, mono=True,
                )
                video_path = _download_video(video_url)

                # Run re-dubbing
                output_path = tempfile.mktemp(suffix=".mp4")
                _run_retalking(svc, video_path, audio_path, output_path)

                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                elapsed = time.time() - start_time
                print(f"video-retalking: generated {len(video_bytes)} bytes in {elapsed:.1f}s")

                return Response(
                    content=video_bytes,
                    media_type="video/mp4",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Model": "video-retalking",
                    },
                )

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                for p in [audio_path, video_path, output_path]:
                    if p and os.path.exists(p):
                        os.unlink(p)

        return api


def _download_video(url: str) -> str:
    """Download video URL to a temp file."""
    import tempfile
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-headers", "User-Agent: Mozilla/5.0",
            "-i", url,
            "-c", "copy",
            path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        if os.path.exists(path):
            os.unlink(path)
        raise RuntimeError(f"Failed to download video: {result.stderr.decode()[:500]}")

    return path


def _run_retalking(svc, video_path: str, audio_path: str, output_path: str):
    """Run video-retalking inference.

    Invokes the video-retalking inference pipeline:
      1. Detect and crop face regions from video frames
      2. Extract facial landmarks and parse face regions
      3. Run DNet/ENet for expression transfer driven by audio
      4. Enhance output faces via GFPGAN
      5. Composite back onto original frames and mux with audio
    """
    import subprocess

    if not getattr(svc, "_model_loaded", False):
        raise RuntimeError("video-retalking model not loaded")

    # Invoke the repo's inference script
    result = subprocess.run(
        [
            "python", f"{RETALKING_DIR}/inference.py",
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
        ],
        capture_output=True,
        cwd=RETALKING_DIR,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )

    if result.returncode != 0:
        stderr = result.stderr.decode()[:1000]
        stdout = result.stdout.decode()[:500]
        raise RuntimeError(
            f"video-retalking inference failed (rc={result.returncode}):\n"
            f"stderr: {stderr}\nstdout: {stdout}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError("video-retalking produced no output file")


@app.local_entrypoint()
def main():
    print("Video-Retalking Lip-Sync Re-Dubbing Service ready")
    print("Deploy with: modal deploy infra/modal-gpu/video_retalking_service.py")
