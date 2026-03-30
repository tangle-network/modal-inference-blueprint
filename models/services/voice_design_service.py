"""
Voice Design Service — Generate synthetic voices from text descriptions.

Wraps Qwen3-TTS VoiceDesign as an HTTP endpoint. Generates voice audio from
natural language descriptions, uploads to R2, returns metadata.

Deploy: modal deploy infra/modal-gpu/voice_design_service.py
Run once: modal run infra/modal-gpu/voice_design_service.py

Endpoint: POST /design
  { "prompt": "Male, 30, warm baritone, Australian", "slug": "...", "seed": 42 }
  → { "slug": "...", "audio_url": "...", "audio_duration_s": 40.2, "audio_hash": "..." }
"""

import modal
import os
import hashlib

from base_service import create_modal_app, create_model_volume, base_service_layer

app = create_modal_app("voice-design")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"

REFERENCE_TEXT = (
    "Hello everyone, welcome back to the show. Today we're going to dive deep into "
    "something really fascinating. I've spent the last few weeks researching this topic, "
    "and honestly, the more I learned, the more surprised I was. So let's start with "
    "the basics. The human voice is one of the most complex instruments in nature. "
    "Every single person has a unique vocal signature, shaped by the size of their "
    "vocal cords, the resonance of their chest cavity, and even the way they learned "
    "to speak as a child."
)


def download_model():
    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    from qwen_tts import Qwen3TTSModel
    Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    print("VoiceDesign model cached.")


image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "soundfile>=0.12.0",
        "numpy<2",
        "huggingface-hub>=0.25.0",
        "qwen-tts>=0.1.0",
        "boto3>=1.34.0",
        "fastapi",
        "uvicorn",
    )
    .run_function(download_model)
)


def upload_to_r2(wav_bytes: bytes, key: str) -> str:
    import boto3
    endpoint = os.environ["S3_ENDPOINT"]
    bucket = os.environ.get("S3_BUCKET", "phony-voice-samples")
    public_url = os.environ.get("S3_PUBLIC_URL", "")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    s3.put_object(Bucket=bucket, Key=key, Body=wav_bytes, ContentType="audio/wav")

    if public_url:
        return f"{public_url}/{key}"
    return f"{endpoint}/{bucket}/{key}"


@app.cls(
    image=image,
    gpu="A10G",
    timeout=180,
    container_idle_timeout=120,
    volumes={MODEL_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("phony-r2")],
    allow_concurrent_inputs=4,
)
class VoiceDesignService:

    @modal.enter()
    def load_model(self):
        import torch
        os.environ["HF_HOME"] = HF_CACHE
        from qwen_tts import Qwen3TTSModel
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("VoiceDesign model loaded on GPU.")

    @modal.asgi_app()
    def web_app(self):
        import torch
        import soundfile as sf
        import io
        import re
        import time
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {"status": "ok", "model": "qwen3-tts-voicedesign-1.7b", "gpu": "A10G"}

        @api.post("/design")
        async def design(request: Request):
            start = time.time()
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            prompt = body.get("prompt", "")
            slug = body.get("slug", "")
            seed = body.get("seed", 42)

            if not prompt:
                return JSONResponse(status_code=400, content={"error": "prompt is required"})
            if not slug:
                slug = re.sub(r'[^a-z0-9]+', '-', prompt.lower())[:60].strip('-') + f"-{int(time.time())}"

            try:
                torch.manual_seed(seed)
                wavs, sr = svc.model.generate_voice_design(
                    text=REFERENCE_TEXT,
                    language="English",
                    instruct=prompt,
                )
                audio = wavs[0]
                duration = len(audio) / sr

                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
                wav_bytes = buf.getvalue()
                audio_hash = hashlib.sha256(wav_bytes).hexdigest()

                audio_url = upload_to_r2(wav_bytes, f"voices/custom/{slug}.wav")

                elapsed = time.time() - start
                print(f"voice-design: {duration:.1f}s audio in {elapsed:.1f}s → {slug}")

                return {
                    "slug": slug,
                    "audio_url": audio_url,
                    "audio_duration_s": round(duration, 2),
                    "audio_hash": audio_hash,
                    "processing_time": round(elapsed, 2),
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        return api


@app.local_entrypoint()
def main():
    print("Voice Design Service")
    print("Deploy: modal deploy infra/modal-gpu/voice_design_service.py")
    print("Endpoint: POST /design { prompt, slug, seed }")
