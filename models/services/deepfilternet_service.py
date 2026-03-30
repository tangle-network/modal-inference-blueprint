"""
DeepFilterNet Speech Enhancement Service - GPU/CPU noise reduction.

Capabilities:
- Real-time speech enhancement / noise suppression
- DeepFilterNet3 model (state-of-the-art DNS challenge)
- Lightweight — runs on CPU or GPU
- Supports any input audio format (via ffmpeg)
- MIT license

Deploy: modal deploy modal/deepfilternet_service.py
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("deepfilternet")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"

# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "deepfilternet",
    )
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="T4",
    timeout=300,
    container_idle_timeout=180,
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=10,
)
class DeepFilterNetService:
    """Speech enhancement via DeepFilterNet3."""

    @modal.enter()
    def load_model(self):
        """Load DeepFilterNet model on container start."""
        from df.enhance import init_df

        self.model, self.df_state, _ = init_df()
        print("DeepFilterNet3 loaded.")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with enhancement endpoint."""
        import time
        import traceback
        import base64
        import io
        import soundfile as sf
        from fastapi import FastAPI, Request
        from fastapi.responses import Response, JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "deepfilternet3",
                "gpu": "T4",
                "capabilities": ["speech_enhancement", "noise_reduction"],
                "sample_rate": svc.df_state.sr(),
            }

        @api.post("/enhance")
        async def enhance(request: Request):
            """Enhance audio by removing noise.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)

            Returns: WAV audio with noise removed.
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

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

                sr = svc.df_state.sr()
                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=sr,
                    mono=True,
                )

                import torch
                from df.enhance import enhance as df_enhance

                audio, _ = sf.read(temp_path, dtype="float32")

                # df.enhance expects (samples,) or (channels, samples)
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                enhanced = df_enhance(svc.model, svc.df_state, audio_tensor)
                enhanced_np = enhanced.squeeze().numpy()

                buf = io.BytesIO()
                sf.write(buf, enhanced_np, sr, format="WAV")
                buf.seek(0)
                wav_data = buf.read()

                elapsed = time.time() - start_time
                duration = len(audio) / sr

                print(
                    f"deepfilternet: enhanced {duration:.1f}s audio "
                    f"in {elapsed:.3f}s (RTF={elapsed / duration:.4f})"
                )

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Processing-Time": f"{elapsed:.3f}",
                        "X-Audio-Duration": f"{duration:.3f}",
                        "X-Sample-Rate": str(sr),
                    },
                )

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
    print("DeepFilterNet Speech Enhancement Service ready")
    print("Deploy with: modal deploy modal/deepfilternet_service.py")
