"""
Spark-TTS Service - LLM-based TTS with BiCodec single-stream speech codec.

Capabilities:
- Zero-shot voice cloning from short reference audio
- Controllable TTS (gender, pitch, speed)
- Built on Qwen2.5 LLM backbone
- BiCodec single-stream decoupled speech tokens
- Apache-2.0 license

Deploy: modal deploy modal/sparktts_service.py

Requirements:
- SparkAudio/Spark-TTS-0.5B model (~1GB)
- A10G GPU (24GB VRAM)
"""

import modal
import os

from base_service import (
    PhonyTTSService,
    create_modal_app,
    create_voice_volume,
    base_service_layer,
)

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("spark-tts")
voice_cache = create_voice_volume("spark-tts")
model_cache = modal.Volume.from_name("phony-spark-tts-model-cache", create_if_missing=True)

MODEL_DIR = "/spark-models"
REPO_DIR = "/spark-tts"

# --- Download model at image build time --------------------------------------

def download_spark_model():
    """Download Spark-TTS model during image build."""
    from huggingface_hub import snapshot_download

    os.makedirs(f"{MODEL_DIR}/Spark-TTS-0.5B", exist_ok=True)
    snapshot_download(
        "SparkAudio/Spark-TTS-0.5B",
        local_dir=f"{MODEL_DIR}/Spark-TTS-0.5B",
    )
    print("Spark-TTS-0.5B download complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "einops==0.8.1",
        "einx==0.3.0",
        "omegaconf==2.3.0",
        "safetensors==0.5.2",
        "soxr==0.5.0.post1",
        "tqdm",
        "transformers==4.46.2",
    )
    .run_commands(
        f"git clone https://github.com/SparkAudio/Spark-TTS.git {REPO_DIR}",
    )
    .run_function(download_spark_model)
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=180,
    volumes={
        "/voice-cache": voice_cache,
        MODEL_DIR: model_cache,
    },
    allow_concurrent_inputs=5,
)
class SparkTTSService(PhonyTTSService):
    """Spark-TTS with zero-shot voice cloning via BiCodec."""

    MODEL_NAME = "spark-tts"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 16000
    VOICE_REF_SR = 16000
    VOICE_REF_MAX_DUR = 15
    FEATURES = ["voice_cloning", "controllable_tts", "llm_backbone"]

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        """Load Spark-TTS model."""
        import sys
        sys.path.insert(0, REPO_DIR)

        from cli.SparkTTS import SparkTTS

        self.tts = SparkTTS(
            model_dir=f"{MODEL_DIR}/Spark-TTS-0.5B",
            device="cuda",
        )
        print("Spark-TTS-0.5B loaded on GPU.")

    def synthesize_impl(self, text: str, voice_id: str, **kwargs) -> tuple:
        import torch

        ref_path = self.get_voice_ref_path(voice_id)
        temperature = kwargs.get("temperature", 0.8)
        top_k = kwargs.get("top_k", 50)
        top_p = kwargs.get("top_p", 0.95)

        with torch.no_grad():
            wav = self.tts.inference(
                text=text,
                prompt_speech_path=ref_path,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        audio_np = wav.squeeze().cpu().numpy()
        return audio_np, self.SAMPLE_RATE

    def synthesize_with_url_impl(self, text: str, ref_audio_path: str, **kwargs) -> tuple:
        """Direct one-shot synthesis from temp reference."""
        import torch

        temperature = kwargs.get("temperature", 0.8)
        top_k = kwargs.get("top_k", 50)
        top_p = kwargs.get("top_p", 0.95)

        with torch.no_grad():
            wav = self.tts.inference(
                text=text,
                prompt_speech_path=ref_audio_path,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        audio_np = wav.squeeze().cpu().numpy()
        return audio_np, self.SAMPLE_RATE

    def extra_routes(self, api):
        """Add controllable TTS endpoint (no voice cloning, style params only)."""
        import time
        import traceback
        from fastapi import Request
        from fastapi.responses import Response, JSONResponse

        svc = self

        @api.post("/synthesize_control")
        async def synthesize_control(request: Request):
            """Synthesize with controllable parameters (no reference audio).

            JSON body:
              - text (str): Text to speak
              - gender (str): "male" or "female"
              - pitch (str): "very_low", "low", "moderate", "high", "very_high"
              - speed (str): "very_low", "low", "moderate", "high", "very_high"
              - temperature (float, optional): default 0.8
              - top_k (int, optional): default 50
              - top_p (float, optional): default 0.95
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            gender = body.get("gender", "female")
            pitch = body.get("pitch", "moderate")
            speed = body.get("speed", "moderate")
            temperature = body.get("temperature", 0.8)
            top_k = body.get("top_k", 50)
            top_p = body.get("top_p", 0.95)

            try:
                import torch

                with torch.no_grad():
                    wav = svc.tts.inference(
                        text=text,
                        prompt_speech_path=None,
                        gender=gender,
                        pitch=pitch,
                        speed=speed,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                audio_np = wav.squeeze().cpu().numpy()

                from base_service import wav_bytes_from_numpy
                wav_data = wav_bytes_from_numpy(audio_np, svc.SAMPLE_RATE)
                elapsed = time.time() - start_time

                print(
                    f"spark-tts: controlled synthesis {len(text)} chars "
                    f"({gender}/{pitch}/{speed}) in {elapsed:.3f}s"
                )

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Sample-Rate": str(svc.SAMPLE_RATE),
                    },
                )
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

    def health_extra(self):
        return {
            "model_version": "0.5B",
            "backbone": "Qwen2.5",
            "codec": "BiCodec",
        }

    @modal.asgi_app()
    def web_app(self):
        return self.build_app()


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Spark-TTS Service ready")
    print("Deploy with: modal deploy modal/sparktts_service.py")
