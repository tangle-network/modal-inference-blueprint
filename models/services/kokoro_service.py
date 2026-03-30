"""
Kokoro-82M TTS Service - Ultra-fast text-to-speech.

Capabilities:
- 54+ voices across 8 languages
- Streaming output (yields chunks)
- ~40-70ms latency on GPU
- Apache 2.0 licensed

Deploy: modal deploy modal/kokoro_service.py
"""

import modal
import os

app = modal.App("phony-kokoro-tts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "ffmpeg")
    .pip_install(
        "kokoro>=0.9.2",
        "soundfile",
        "torch",
        "numpy<2",
        "fastapi",
        "uvicorn",
    )
)


@app.cls(
    image=image,
    gpu="T4",  # Kokoro is lightweight, T4 is sufficient
    timeout=300,
    container_idle_timeout=120,  # Keep warm for 2 minutes
    allow_concurrent_inputs=10,
)
class KokoroService:
    """Kokoro TTS service with streaming support."""

    @modal.enter()
    def load_model(self):
        """Load Kokoro pipeline on startup."""
        from kokoro import KPipeline

        # Default to American English
        self.pipeline = KPipeline(lang_code='a')
        print("Kokoro pipeline loaded")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with TTS endpoints."""
        from fastapi import FastAPI, Request
        from fastapi.responses import Response, JSONResponse, StreamingResponse
        import soundfile as sf
        import io
        import time

        api = FastAPI()

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "kokoro-tts",
                "model": "Kokoro-82M",
                "voices": 54,
                "languages": 8
            }

        @api.get("/voices")
        def list_voices():
            """List available voices."""
            # Kokoro voice naming: {lang}_{style}_{name}
            # See: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
            return {
                "voices": [
                    # American English
                    {"id": "af_heart", "name": "Heart (Female)", "lang": "en-us"},
                    {"id": "af_bella", "name": "Bella (Female)", "lang": "en-us"},
                    {"id": "af_nicole", "name": "Nicole (Female)", "lang": "en-us"},
                    {"id": "af_sarah", "name": "Sarah (Female)", "lang": "en-us"},
                    {"id": "af_sky", "name": "Sky (Female)", "lang": "en-us"},
                    {"id": "am_adam", "name": "Adam (Male)", "lang": "en-us"},
                    {"id": "am_michael", "name": "Michael (Male)", "lang": "en-us"},
                    # British English
                    {"id": "bf_emma", "name": "Emma (Female)", "lang": "en-gb"},
                    {"id": "bf_isabella", "name": "Isabella (Female)", "lang": "en-gb"},
                    {"id": "bm_george", "name": "George (Male)", "lang": "en-gb"},
                    {"id": "bm_lewis", "name": "Lewis (Male)", "lang": "en-gb"},
                    # Add more as needed...
                ],
                "default": "af_heart"
            }

        @api.post("/synthesize")
        async def synthesize(request: Request):
            """Synthesize speech from text.

            Request body:
            {
                "text": "Text to synthesize",
                "voice": "af_heart",  // optional, defaults to af_heart
                "speed": 1.0  // optional, speech speed multiplier
            }

            Returns: WAV audio
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice = body.get("voice", "af_heart")
            speed = body.get("speed", 1.0)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            try:
                # Generate audio
                audio_chunks = []
                generator = self.pipeline(text, voice=voice, speed=speed)

                for i, (gs, ps, audio) in enumerate(generator):
                    audio_chunks.append(audio)

                # Concatenate all chunks
                import numpy as np
                full_audio = np.concatenate(audio_chunks)

                # Write to WAV buffer
                buffer = io.BytesIO()
                sf.write(buffer, full_audio, 24000, format='WAV')
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"Synthesized {len(text)} chars in {elapsed:.3f}s")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Voice": voice
                    }
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/synthesize/stream")
        async def synthesize_stream(request: Request):
            """Stream synthesized speech chunk by chunk.

            Request body:
            {
                "text": "Text to synthesize",
                "voice": "af_heart",
                "speed": 1.0
            }

            Returns: Streaming WAV chunks
            """
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice = body.get("voice", "af_heart")
            speed = body.get("speed", 1.0)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            def generate():
                """Generator that yields audio chunks."""
                import numpy as np

                generator = self.pipeline(text, voice=voice, speed=speed)

                first_chunk = True
                for i, (gs, ps, audio) in enumerate(generator):
                    # Convert to bytes
                    buffer = io.BytesIO()
                    sf.write(buffer, audio, 24000, format='WAV')
                    buffer.seek(0)

                    if first_chunk:
                        # Include WAV header for first chunk
                        yield buffer.read()
                        first_chunk = False
                    else:
                        # Skip WAV header for subsequent chunks (raw PCM)
                        buffer.seek(44)  # WAV header is 44 bytes
                        yield buffer.read()

            return StreamingResponse(
                generate(),
                media_type="audio/wav",
                headers={"X-Voice": voice}
            )

        return api


# For local testing
@app.local_entrypoint()
def main():
    """Test the service locally."""
    print("Kokoro TTS service ready")
    print("Deploy with: modal deploy modal/kokoro_service.py")
