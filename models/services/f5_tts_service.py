"""
F5-TTS Service - Zero-shot voice cloning from ~10s of reference audio.

Capabilities:
- Zero-shot voice cloning (no training, no enrollment)
- ~200-400ms TTFB on A10G
- High-quality natural speech
- Apache 2.0 licensed (SWivid/F5-TTS)

Deploy: modal deploy modal/f5_tts_service.py
"""

import modal
import os

app = modal.App("phony-f5-tts")

voice_cache = modal.Volume.from_name("phony-f5-voices", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "espeak-ng")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "transformers",
        "cached_path",
        "jieba",
        "pypinyin",
        "tomli",
        "vocos",
    )
    .run_commands(
        "pip install git+https://github.com/SWivid/F5-TTS.git"
    )
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=120,
    volumes={"/voice-cache": voice_cache},
    allow_concurrent_inputs=5,
)
class F5TTSService:
    """F5-TTS service with zero-shot voice cloning."""

    @modal.enter()
    def load_model(self):
        """Load F5-TTS model on startup."""
        from f5_tts.api import F5TTS

        self.model = F5TTS()
        print("F5-TTS model loaded")

        os.makedirs("/voice-cache/references", exist_ok=True)

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with voice cloning TTS endpoints."""
        from fastapi import FastAPI, Request, UploadFile, File, Form
        from fastapi.responses import Response, JSONResponse, StreamingResponse
        import soundfile as sf
        import io
        import time
        import tempfile
        import subprocess

        api = FastAPI()

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "f5-tts",
                "model": "F5-TTS",
                "features": ["zero_shot_cloning", "streaming"],
            }

        @api.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            """Register a voice reference for zero-shot cloning.

            F5-TTS doesn't need training — it just stores the reference audio
            and uses it at synthesis time for zero-shot cloning.

            Recommended: 10-15 seconds of clear speech, 24kHz mono WAV.
            """
            if not audio_file and not audio_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Either audio_file or audio_url required"},
                )

            try:
                if audio_url:
                    import requests as req

                    resp = req.get(audio_url, timeout=30)
                    audio_data = resp.content
                else:
                    audio_data = await audio_file.read()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    f.write(audio_data)

                ref_path = f"/voice-cache/references/{voice_id}.wav"
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", temp_path,
                        "-ar", "24000", "-ac", "1", "-f", "wav", ref_path,
                    ],
                    capture_output=True,
                )
                os.unlink(temp_path)

                if result.returncode != 0:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Failed to process audio"},
                    )

                voice_cache.commit()

                return {
                    "status": "success",
                    "voice_id": voice_id,
                    "message": f"Voice '{voice_id}' registered for zero-shot cloning",
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/register_voice")
        async def register_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            """Alias for /clone_voice (matches Pocket TTS / Chatterbox API)."""
            return await clone_voice(voice_id=voice_id, audio_file=audio_file, audio_url=audio_url)

        @api.get("/voices")
        def list_voices():
            """List registered voice references."""
            voices = []
            ref_dir = "/voice-cache/references"
            if os.path.exists(ref_dir):
                for f in os.listdir(ref_dir):
                    if f.endswith(".wav"):
                        voices.append({"id": f[:-4], "type": "cloned"})
            return {"voices": voices}

        @api.delete("/voices/{voice_id}")
        def delete_voice(voice_id: str):
            """Delete a cloned voice."""
            ref_path = f"/voice-cache/references/{voice_id}.wav"
            if os.path.exists(ref_path):
                os.unlink(ref_path)
                voice_cache.commit()
                return {"status": "deleted", "voice_id": voice_id}
            return JSONResponse(status_code=404, content={"error": "Voice not found"})

        @api.post("/synthesize")
        async def synthesize(request: Request):
            """Synthesize speech with zero-shot voice cloning.

            Request body:
            {
                "text": "Text to synthesize",
                "voice_id": "huberman",
                "speed": 1.0
            }

            Returns: WAV audio (24kHz mono)
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")
            speed = body.get("speed", 1.0)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            ref_path = f"/voice-cache/references/{voice_id}.wav"
            if not os.path.exists(ref_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found. Register it first with /clone_voice"},
                )

            try:
                # F5-TTS infer: reference audio + reference text (empty for auto) + target text
                wav, sr, _ = self.model.infer(
                    ref_file=ref_path,
                    ref_text="",  # Auto-transcribe reference
                    gen_text=text,
                    speed=speed,
                )

                buffer = io.BytesIO()
                sf.write(buffer, wav, sr, format="WAV")
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"F5-TTS synthesized {len(text)} chars with voice '{voice_id}' in {elapsed:.3f}s")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Voice-Id": voice_id,
                        "X-Sample-Rate": str(sr),
                    },
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/synthesize_stream")
        async def synthesize_stream(request: Request):
            """Stream synthesized speech by splitting into sentences."""
            import re
            import asyncio

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")
            speed = body.get("speed", 1.0)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            ref_path = f"/voice-cache/references/{voice_id}.wav"
            if not os.path.exists(ref_path):
                return JSONResponse(status_code=404, content={"error": f"Voice '{voice_id}' not found"})

            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return JSONResponse(status_code=400, content={"error": "No sentences found"})

            model = self.model

            def synthesize_sentence(sentence: str) -> bytes:
                wav, sr, _ = model.infer(
                    ref_file=ref_path,
                    ref_text="",
                    gen_text=sentence,
                    speed=speed,
                )
                buffer = io.BytesIO()
                sf.write(buffer, wav, sr, format="WAV")
                buffer.seek(0)
                return buffer.read()

            async def stream_audio():
                loop = asyncio.get_event_loop()
                for i, sentence in enumerate(sentences):
                    start = time.time()
                    audio_data = await loop.run_in_executor(None, synthesize_sentence, sentence)
                    elapsed = time.time() - start
                    print(f"  F5-TTS sentence {i+1}/{len(sentences)}: {len(sentence)} chars in {elapsed:.2f}s")
                    yield audio_data

            return StreamingResponse(
                stream_audio(),
                media_type="audio/wav",
                headers={"X-Sentences": str(len(sentences))},
            )

        @api.post("/synthesize_with_url")
        async def synthesize_with_url(request: Request):
            """One-shot cloning: synthesize using a reference audio URL (no pre-registration)."""
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            speaker_wav_url = body.get("speaker_wav_url")
            speed = body.get("speed", 1.0)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not speaker_wav_url:
                return JSONResponse(status_code=400, content={"error": "speaker_wav_url is required"})

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-headers", "User-Agent: Mozilla/5.0",
                        "-i", speaker_wav_url,
                        "-ar", "24000", "-ac", "1", "-t", "30",
                        "-f", "wav", temp_path,
                    ],
                    capture_output=True,
                )

                if result.returncode != 0:
                    return JSONResponse(status_code=400, content={"error": "Failed to download/convert audio"})

                wav, sr, _ = self.model.infer(
                    ref_file=temp_path,
                    ref_text="",
                    gen_text=text,
                    speed=speed,
                )
                os.unlink(temp_path)

                buffer = io.BytesIO()
                sf.write(buffer, wav, sr, format="WAV")
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"F5-TTS one-shot synthesis in {elapsed:.3f}s")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={"X-Generation-Time": f"{elapsed:.3f}"},
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        return api


@app.local_entrypoint()
def main():
    print("F5-TTS service ready")
    print("Deploy with: modal deploy modal/f5_tts_service.py")
