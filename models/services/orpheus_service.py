"""
Orpheus TTS Service — Llama-3B backbone speech synthesis on Modal.

Capabilities:
- 8 preset voices: tara, leah, jess, leo, dan, mia, zac, zoe
- Emotion control via inline tags (<laugh>, <sigh>, <chuckle>, etc.)
- ~200ms streaming latency
- 24kHz output
- Apache 2.0 license (Canopy Labs)

NOTE: Orpheus uses preset voices, not voice cloning from audio references.
The /clone_voice and /synthesize_with_url endpoints return 501 (not supported).
Voice selection is done via the `voice` parameter on /synthesize.

Deploy: modal deploy modal/orpheus_service.py
"""

import modal
import os
import io
import time
import struct
import traceback

from base_service import (
    base_service_layer,
    ModalTTSService,
    create_modal_app,
    wav_bytes_from_numpy,
)

app = create_modal_app("orpheus-tts")

model_cache = modal.Volume.from_name("phony-model-cache", create_if_missing=True)

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "numpy<2",
        "scipy",
        "fastapi",
        "uvicorn",
        "soundfile",
        "vllm==0.7.3",
        "orpheus-speech",
    )
)

AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

EMOTION_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>"]

DEFAULT_VOICE = "tara"


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=180,
    volumes={"/model-cache": model_cache},
    allow_concurrent_inputs=5,
)
class OrpheusService(ModalTTSService):
    """Orpheus TTS service with preset voices and emotion control."""

    MODEL_NAME = "orpheus-tts"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 24000
    FEATURES = ["preset_voices", "emotion_tags", "streaming"]
    CONTAINER_IDLE = 180
    CONCURRENT_INPUTS = 5

    @modal.enter()
    def load_model(self):
        os.makedirs("/model-cache/orpheus", exist_ok=True)
        self.setup_model()

    def setup_model(self):
        os.environ.setdefault("HF_HOME", "/model-cache/orpheus")

        from orpheus_tts import OrpheusModel

        self.model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
            max_model_len=2048,
        )
        print("Orpheus TTS model loaded (canopylabs/orpheus-tts-0.1-finetune-prod)")

    def _collect_audio(self, chunk_generator) -> bytes:
        """Collect PCM int16 chunks from generate_speech into a contiguous buffer."""
        pcm_data = b""
        for chunk in chunk_generator:
            if chunk is not None:
                pcm_data += chunk
        return pcm_data

    def _pcm_to_numpy(self, pcm_bytes: bytes):
        """Convert raw int16 PCM bytes to float32 numpy array."""
        import numpy as np
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

    def synthesize_impl(self, text, voice_id, **kwargs):
        voice = kwargs.get("voice", voice_id)
        if voice not in AVAILABLE_VOICES:
            voice = DEFAULT_VOICE

        temperature = kwargs.get("temperature", 0.6)
        top_p = kwargs.get("top_p", 0.8)
        max_tokens = kwargs.get("max_tokens", 1200)
        repetition_penalty = kwargs.get("repetition_penalty", 1.3)

        chunk_gen = self.model.generate_speech(
            prompt=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )

        pcm_data = self._collect_audio(chunk_gen)
        if not pcm_data:
            import numpy as np
            return np.zeros(0, dtype=np.float32), self.SAMPLE_RATE

        audio_np = self._pcm_to_numpy(pcm_data)
        return audio_np, self.SAMPLE_RATE

    def synthesize_stream_impl(self, text, voice_id, **kwargs):
        """Yield audio chunks as they arrive from Orpheus streaming."""
        voice = kwargs.get("voice", voice_id)
        if voice not in AVAILABLE_VOICES:
            voice = DEFAULT_VOICE

        temperature = kwargs.get("temperature", 0.6)
        top_p = kwargs.get("top_p", 0.8)
        max_tokens = kwargs.get("max_tokens", 1200)
        repetition_penalty = kwargs.get("repetition_penalty", 1.3)

        chunk_gen = self.model.generate_speech(
            prompt=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )

        for chunk in chunk_gen:
            if chunk is not None and len(chunk) > 0:
                audio_np = self._pcm_to_numpy(chunk)
                yield audio_np, self.SAMPLE_RATE

    # -- Orpheus uses preset voices, not audio-reference cloning ---------------

    def voice_exists(self, voice_id: str) -> bool:
        """Preset voices are always available."""
        return voice_id in AVAILABLE_VOICES

    def list_voice_ids(self) -> list[str]:
        return list(AVAILABLE_VOICES)

    def health_extra(self):
        return {
            "voices": AVAILABLE_VOICES,
            "emotion_tags": EMOTION_TAGS,
            "model": "canopylabs/orpheus-tts-0.1-finetune-prod",
            "version": "v1",
        }

    def extra_routes(self, api):
        from fastapi import Request
        from fastapi.responses import Response, JSONResponse, StreamingResponse

        svc = self

        # Override /synthesize to use `voice` param instead of voice_id ref lookup
        @api.post("/synthesize")
        async def synthesize(request: Request):
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice = body.get("voice") or body.get("voice_id") or DEFAULT_VOICE

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            if voice not in AVAILABLE_VOICES:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown voice '{voice}'. Available: {AVAILABLE_VOICES}",
                    },
                )

            extra = {
                k: v for k, v in body.items()
                if k not in ("text", "voice", "voice_id")
            }

            try:
                audio_np, sr = svc.synthesize_impl(text, voice, voice=voice, **extra)
                wav_data = wav_bytes_from_numpy(audio_np, sr)
                elapsed = time.time() - start_time

                print(f"orpheus-tts: synthesized {len(text)} chars with '{voice}' in {elapsed:.3f}s")

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Voice-Id": voice,
                        "X-Sample-Rate": str(sr),
                    },
                )
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        # Override /synthesize_stream for voice param
        @api.post("/synthesize_stream")
        async def synthesize_stream(request: Request):
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice = body.get("voice") or body.get("voice_id") or DEFAULT_VOICE

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            if voice not in AVAILABLE_VOICES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unknown voice '{voice}'. Available: {AVAILABLE_VOICES}"},
                )

            extra = {
                k: v for k, v in body.items()
                if k not in ("text", "voice", "voice_id")
            }

            def generate():
                first = True
                for audio_np, sr in svc.synthesize_stream_impl(text, voice, voice=voice, **extra):
                    data = wav_bytes_from_numpy(audio_np, sr)
                    if first:
                        yield data
                        first = False
                    else:
                        yield data[44:]  # skip WAV header on subsequent chunks

            return StreamingResponse(
                generate(),
                media_type="audio/wav",
                headers={"X-Voice-Id": voice},
            )

        # Override /voices to return preset voice metadata
        @api.get("/voices")
        def list_voices():
            voice_meta = {
                "tara": {"gender": "female", "style": "conversational, clear"},
                "leah": {"gender": "female", "style": "warm, gentle"},
                "jess": {"gender": "female", "style": "energetic, youthful"},
                "leo": {"gender": "male", "style": "authoritative, deep"},
                "dan": {"gender": "male", "style": "friendly, casual"},
                "mia": {"gender": "female", "style": "professional, articulate"},
                "zac": {"gender": "male", "style": "enthusiastic, dynamic"},
                "zoe": {"gender": "female", "style": "calm, soothing"},
            }
            voices = [
                {"id": v, "type": "preset", **voice_meta.get(v, {})}
                for v in AVAILABLE_VOICES
            ]
            return {"voices": voices}

        # Disable cloning endpoints (not supported by Orpheus)
        @api.post("/clone_voice")
        async def clone_voice_unsupported():
            return JSONResponse(
                status_code=501,
                content={"error": "Orpheus TTS uses preset voices. Voice cloning from audio is not supported."},
            )

        @api.post("/register_voice")
        async def register_voice_unsupported():
            return JSONResponse(
                status_code=501,
                content={"error": "Orpheus TTS uses preset voices. Voice cloning from audio is not supported."},
            )

        @api.post("/synthesize_with_url")
        async def synthesize_with_url_unsupported():
            return JSONResponse(
                status_code=501,
                content={"error": "Orpheus TTS uses preset voices. One-shot voice cloning is not supported."},
            )

    @modal.asgi_app()
    def web_app(self):
        return self.build_app()


@app.local_entrypoint()
def main():
    print("Orpheus TTS service ready")
    print(f"Available voices: {', '.join(AVAILABLE_VOICES)}")
    print(f"Emotion tags: {', '.join(EMOTION_TAGS)}")
    print("Deploy with: modal deploy modal/orpheus_service.py")
