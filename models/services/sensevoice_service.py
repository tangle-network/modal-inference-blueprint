"""
SenseVoice STT Service - ASR + emotion recognition + audio event detection.

Capabilities:
- Multilingual ASR (50+ languages including zh, en, yue, ja, ko)
- Speech emotion recognition (happy, sad, angry, neutral)
- Audio event detection (laughter, applause, music, crying, coughing)
- 5x faster than Whisper-Small, 15x faster on SenseVoice-Small
- Alibaba FunAudioLLM, Apache-2.0 license

Deploy: modal deploy modal/sensevoice_service.py

Requirements:
- funasr >= 1.1.3
- A10G GPU (24GB VRAM)
"""

import modal
import os

from base_service import create_modal_app, create_model_volume, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("sensevoice")
model_cache = create_model_volume()

MODEL_CACHE_DIR = "/models"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"

# --- Download model at image build time --------------------------------------

def download_sensevoice():
    """Download SenseVoice model during image build."""
    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE

    from funasr import AutoModel

    AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        remote_code="./model.py",
        device="cpu",
    )
    print("SenseVoice-Small download complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.10"))
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "funasr>=1.1.3",
        "modelscope",
        "huggingface_hub",
    )
    .run_function(download_sensevoice)
)


# --- Emotion / event post-processing ----------------------------------------

EMOTION_MAP = {
    "<|HAPPY|>": "happy",
    "<|SAD|>": "sad",
    "<|ANGRY|>": "angry",
    "<|NEUTRAL|>": "neutral",
}

EVENT_TAGS = {
    "<|BGM|>": "music",
    "<|Speech|>": "speech",
    "<|Applause|>": "applause",
    "<|Laughter|>": "laughter",
    "<|Cry|>": "crying",
    "<|Sneeze|>": "sneeze",
    "<|Breathe|>": "breathing",
    "<|Cough|>": "cough",
}


def parse_sensevoice_output(raw_text: str) -> dict:
    """Parse SenseVoice rich transcription output into structured fields."""
    import re

    emotions = []
    events = []
    clean_text = raw_text

    for tag, label in EMOTION_MAP.items():
        if tag in raw_text:
            emotions.append(label)
            clean_text = clean_text.replace(tag, "")

    for tag, label in EVENT_TAGS.items():
        if tag in raw_text:
            events.append(label)
            clean_text = clean_text.replace(tag, "")

    # Remove any remaining special tokens
    clean_text = re.sub(r"<\|[^|]+\|>", "", clean_text).strip()

    return {
        "text": clean_text,
        "emotions": emotions,
        "events": events,
        "raw": raw_text,
    }


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=300,
    volumes={MODEL_CACHE_DIR: model_cache},
    allow_concurrent_inputs=10,
)
class SenseVoiceService:
    """Speech-to-text + emotion + audio events via SenseVoice-Small."""

    @modal.enter()
    def load_model(self):
        """Load SenseVoice model on container start."""
        os.environ["HF_HOME"] = HF_CACHE

        from funasr import AutoModel

        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            remote_code="./model.py",
            device="cuda:0",
        )
        print("SenseVoice-Small loaded on GPU.")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with transcription + emotion + event detection."""
        import time
        import traceback
        import base64
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "sensevoice-small",
                "gpu": "A10G",
                "capabilities": [
                    "transcribe", "emotion_recognition",
                    "audio_event_detection", "multilingual",
                ],
                "languages": "50+",
                "supported_emotions": list(EMOTION_MAP.values()),
                "supported_events": list(EVENT_TAGS.values()),
            }

        @api.post("/transcribe")
        async def transcribe(request: Request):
            """Transcribe audio with emotion recognition and event detection.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - language (str, optional): language hint ("auto", "zh", "en", "yue", "ja", "ko")
              - use_itn (bool, optional): inverse text normalization (default true)

            Returns:
              {
                "text": "transcription without tags",
                "language": "en",
                "emotions": ["happy"],
                "events": ["speech"],
                "raw": "raw output with tags",
                "duration_seconds": 10.5,
                "processing_time": 0.3
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")
            language = body.get("language", "auto")
            use_itn = body.get("use_itn", True)

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

                res = svc.model.generate(
                    input=temp_path,
                    cache={},
                    language=language,
                    use_itn=use_itn,
                    batch_size_s=0,
                )

                raw_text = ""
                if res and len(res) > 0:
                    if isinstance(res[0], dict):
                        raw_text = res[0].get("text", "")
                    elif hasattr(res[0], "text"):
                        raw_text = res[0].text
                    else:
                        raw_text = str(res[0])

                parsed = parse_sensevoice_output(raw_text)

                import soundfile as sf
                info = sf.info(temp_path)
                duration = info.duration

                elapsed = time.time() - start_time

                result = {
                    "text": parsed["text"],
                    "emotions": parsed["emotions"],
                    "events": parsed["events"],
                    "raw": parsed["raw"],
                    "duration_seconds": round(duration, 3),
                    "processing_time": round(elapsed, 3),
                }

                if language != "auto":
                    result["language"] = language

                rtf = elapsed / duration if duration > 0 else 0
                print(
                    f"sensevoice: transcribed {duration:.1f}s audio "
                    f"in {elapsed:.3f}s (RTF={rtf:.4f}), "
                    f"emotions={parsed['emotions']}, events={parsed['events']}"
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
    print("SenseVoice STT Service ready")
    print("Deploy with: modal deploy modal/sensevoice_service.py")
