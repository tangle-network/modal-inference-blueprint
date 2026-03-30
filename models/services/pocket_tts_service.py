"""
Pocket TTS Service - Lightweight CPU-based voice cloning.

Capabilities:
- 100M parameters, runs real-time on CPU
- Voice cloning from ~5 seconds of audio
- ~200ms first-audio latency
- MIT licensed

Deploy: modal deploy modal/pocket_tts_service.py
"""

import modal
import os

app = modal.App("phony-pocket-tts")

# Volume for storing cloned voice references
voice_cache = modal.Volume.from_name("phony-pocket-voices", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "pocket-tts",
        "numpy<2",
        "scipy",
        "fastapi",
        "uvicorn",
        "soundfile",
        "requests",
    )
)


@app.cls(
    image=image,
    cpu=4,  # Pocket TTS runs on CPU - allocate multiple cores for parallel processing
    memory=8192,  # 8GB RAM for model and audio processing
    timeout=600,
    container_idle_timeout=180,  # Keep warm for 3 minutes
    volumes={"/voice-cache": voice_cache},
    allow_concurrent_inputs=5,
)
class PocketTTSService:
    """Pocket TTS service with voice cloning - CPU only."""

    @modal.enter()
    def load_model(self):
        """Load Pocket TTS model on startup."""
        from pocket_tts import TTSModel

        # Load the model (downloads weights on first run)
        self.model = TTSModel.load_model()
        self.sample_rate = self.model.sample_rate
        print(f"Pocket TTS model loaded (sample_rate={self.sample_rate})")

        # Create voice cache directory
        os.makedirs("/voice-cache/references", exist_ok=True)

        # Cache voice states for faster synthesis
        self._voice_states: dict = {}
        self._warmed_voices: set[str] = set()

        # Built-in voices
        self._builtin_voices = ["alba", "marius", "javert"]

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with voice cloning TTS endpoints."""
        from fastapi import FastAPI, Request, UploadFile, File, Form
        from fastapi.responses import Response, JSONResponse
        import soundfile as sf
        import io
        import time
        import requests
        import tempfile

        api = FastAPI()

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "pocket-tts",
                "model": "Pocket-TTS-100M",
                "version": "v1",
                "sample_rate": self.sample_rate,
                "features": ["voice_cloning", "cpu_inference", "low_latency", "streaming"],
                "builtin_voices": self._builtin_voices,
                "compute": "cpu"
            }

        @api.get("/info")
        def info():
            """Return detailed service information."""
            return {
                "service": "pocket-tts",
                "model": "Pocket-TTS",
                "parameters": "100M",
                "compute": "cpu",
                "latency": "~200ms first-audio",
                "voice_cloning": {
                    "min_audio_duration": "5 seconds",
                    "recommended_duration": "10-15 seconds",
                    "supported_formats": ["wav", "mp3", "m4a", "ogg", "flac"]
                },
                "license": "MIT"
            }

        @api.get("/voices")
        def list_voices():
            """List all available voices (builtin + cloned)."""
            voices = []

            # Add builtin voices
            for v in self._builtin_voices:
                voices.append({
                    "id": v,
                    "type": "builtin",
                    "warmed": v in self._warmed_voices
                })

            # Add cloned voices
            ref_dir = "/voice-cache/references"
            if os.path.exists(ref_dir):
                for f in os.listdir(ref_dir):
                    if f.endswith(".wav"):
                        voice_id = f[:-4]
                        ref_path = os.path.join(ref_dir, f)
                        file_size = os.path.getsize(ref_path)
                        voices.append({
                            "id": voice_id,
                            "type": "cloned",
                            "size_bytes": file_size,
                            "warmed": voice_id in self._warmed_voices
                        })
            return {"voices": voices, "count": len(voices)}

        @api.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None)
        ):
            """Register a voice for cloning from audio sample.

            Either upload audio_file or provide audio_url.
            Recommended: 5-15 seconds of clear speech.
            """
            if not audio_file and not audio_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Either audio_file or audio_url required"}
                )

            # Validate voice_id
            if not voice_id or not voice_id.strip():
                return JSONResponse(
                    status_code=400,
                    content={"error": "voice_id cannot be empty"}
                )

            # Sanitize voice_id (alphanumeric and underscores only)
            sanitized_id = "".join(c for c in voice_id if c.isalnum() or c == "_")
            if sanitized_id != voice_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "voice_id must contain only alphanumeric characters and underscores"}
                )

            try:
                # Get audio data
                if audio_url:
                    # Download from URL
                    try:
                        response = requests.get(audio_url, timeout=30)
                        response.raise_for_status()
                        audio_data = response.content
                    except requests.RequestException as e:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Failed to download audio: {str(e)}"}
                        )
                else:
                    audio_data = await audio_file.read()

                if len(audio_data) < 1000:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Audio file too small - need at least 5 seconds of audio"}
                    )

                # Save to temporary file for processing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    f.write(audio_data)

                # Convert to mono WAV (Pocket TTS handles resampling internally)
                import subprocess
                ref_path = f"/voice-cache/references/{voice_id}.wav"
                result = subprocess.run([
                    "ffmpeg", "-y", "-i", temp_path,
                    "-ac", "1", "-f", "wav", ref_path
                ], capture_output=True)

                os.unlink(temp_path)

                if result.returncode != 0:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Failed to process audio",
                            "details": result.stderr.decode()[:500]
                        }
                    )

                # Verify the file was created
                if not os.path.exists(ref_path):
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Failed to save voice reference"}
                    )

                # Get audio duration for feedback
                try:
                    info_result = subprocess.run([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", ref_path
                    ], capture_output=True, text=True)
                    duration = float(info_result.stdout.strip())
                except Exception:
                    duration = None

                # Commit volume changes
                voice_cache.commit()

                # Clear from warmed set since it's a new/updated voice
                self._warmed_voices.discard(voice_id)

                return {
                    "status": "success",
                    "voice_id": voice_id,
                    "message": f"Voice '{voice_id}' registered for cloning",
                    "duration_seconds": duration
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/warmup")
        async def warmup(request: Request):
            """Warm up a voice by pre-loading its state.

            Call this when a chat session starts to minimize first synthesis latency.
            """
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

            voice_id = body.get("voice_id")
            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            # Check if it's a builtin or cloned voice
            if voice_id in self._builtin_voices:
                audio_prompt = voice_id
            else:
                ref_path = f"/voice-cache/references/{voice_id}.wav"
                if not os.path.exists(ref_path):
                    return JSONResponse(status_code=404, content={"error": f"Voice '{voice_id}' not found"})
                audio_prompt = ref_path

            # Warm up by loading voice state and doing a short synthesis
            start = time.time()
            try:
                voice_state = self.model.get_state_for_audio_prompt(audio_prompt)
                self._voice_states[voice_id] = voice_state
                # Do a short synthesis to warm GPU/CPU caches
                _ = self.model.generate_audio(voice_state, "Hello.")
                elapsed = time.time() - start
                self._warmed_voices.add(voice_id)
                print(f"Warmed up voice '{voice_id}' in {elapsed:.2f}s")
                return {"status": "warmed", "voice_id": voice_id, "warmup_time": elapsed}
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        def get_voice_state(voice_id: str):
            """Get or create voice state for a voice ID."""
            # Return cached state if available
            if voice_id in self._voice_states:
                return self._voice_states[voice_id]

            # Determine audio prompt (builtin name or file path)
            if voice_id in self._builtin_voices:
                audio_prompt = voice_id
            else:
                ref_path = f"/voice-cache/references/{voice_id}.wav"
                if not os.path.exists(ref_path):
                    return None
                audio_prompt = ref_path

            # Load and cache voice state
            voice_state = self.model.get_state_for_audio_prompt(audio_prompt)
            self._voice_states[voice_id] = voice_state
            return voice_state

        @api.post("/synthesize")
        async def synthesize(request: Request):
            """Synthesize speech with voice cloning.

            Request body:
            {
                "text": "Text to synthesize",
                "voice_id": "huberman"  // ID of builtin or cloned voice
            }

            Returns: WAV audio
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            # Get voice state
            voice_state = get_voice_state(voice_id)
            if voice_state is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found. Clone it first with /clone_voice or use builtin: {self._builtin_voices}"}
                )

            try:
                load_time = time.time()

                # Generate speech with voice state
                wav = self.model.generate_audio(voice_state, text)

                gen_time = time.time() - load_time

                # Handle output format (numpy array or tensor)
                import numpy as np
                if hasattr(wav, 'numpy'):
                    audio_np = wav.numpy()
                elif hasattr(wav, 'cpu'):
                    audio_np = wav.squeeze().cpu().numpy()
                else:
                    audio_np = np.array(wav)

                # Ensure 1D array
                audio_np = audio_np.squeeze()

                # Write to WAV buffer
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, self.sample_rate, format='WAV')
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"Synthesized {len(text)} chars with voice '{voice_id}' in {elapsed:.3f}s (gen: {gen_time:.3f}s)")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Inference-Time": f"{gen_time:.3f}",
                        "X-Voice-Id": voice_id,
                        "X-Sample-Rate": str(self.sample_rate),
                        "X-Compute": "cpu"
                    }
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/synthesize_stream")
        async def synthesize_stream(request: Request):
            """Synthesize speech with streaming response.

            Same as /synthesize but returns chunked audio for lower time-to-first-byte.
            Note: Pocket TTS generates full audio at once, so this streams the result in chunks.
            """
            from fastapi.responses import StreamingResponse

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            voice_state = get_voice_state(voice_id)
            if voice_state is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found"}
                )

            async def audio_generator():
                """Generate audio and stream in chunks."""
                import numpy as np

                # Generate full audio
                wav = self.model.generate_audio(voice_state, text)

                if hasattr(wav, 'numpy'):
                    audio_np = wav.numpy()
                elif hasattr(wav, 'cpu'):
                    audio_np = wav.squeeze().cpu().numpy()
                else:
                    audio_np = np.array(wav)

                audio_np = audio_np.squeeze()

                # Write to WAV buffer
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, self.sample_rate, format='WAV')
                buffer.seek(0)

                # Stream in 8KB chunks
                chunk_size = 8192
                while True:
                    chunk = buffer.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            return StreamingResponse(
                audio_generator(),
                media_type="audio/wav",
                headers={"X-Voice-Id": voice_id, "X-Compute": "cpu"}
            )

        @api.post("/synthesize_with_url")
        async def synthesize_with_url(request: Request):
            """One-shot voice cloning from URL (no pre-registration needed).

            Request body:
            {
                "text": "Text to synthesize",
                "speaker_wav_url": "https://..."  // URL to voice sample
            }

            Returns: WAV audio
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            speaker_wav_url = body.get("speaker_wav_url")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            if not speaker_wav_url:
                return JSONResponse(status_code=400, content={"error": "speaker_wav_url is required"})

            try:
                # Download and convert voice sample
                import subprocess

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                result = subprocess.run([
                    "ffmpeg", "-y",
                    "-headers", "User-Agent: Mozilla/5.0",
                    "-i", speaker_wav_url,
                    "-ar", "24000", "-ac", "1", "-t", "30",  # Max 30 seconds
                    "-f", "wav", temp_path
                ], capture_output=True)

                if result.returncode != 0:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Failed to download/convert audio"}
                    )

                load_time = time.time()

                # Get voice state from audio file
                voice_state = self.model.get_state_for_audio_prompt(temp_path)

                # Generate speech
                wav = self.model.generate_audio(voice_state, text)
                os.unlink(temp_path)

                gen_time = time.time() - load_time

                # Handle output format
                import numpy as np
                if hasattr(wav, 'numpy'):
                    audio_np = wav.numpy()
                elif hasattr(wav, 'cpu'):
                    audio_np = wav.squeeze().cpu().numpy()
                else:
                    audio_np = np.array(wav)

                audio_np = audio_np.squeeze()

                # Write to WAV buffer
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, self.sample_rate, format='WAV')
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"One-shot synthesis in {elapsed:.3f}s (gen: {gen_time:.3f}s)")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Inference-Time": f"{gen_time:.3f}",
                        "X-Sample-Rate": str(self.sample_rate),
                        "X-Compute": "cpu"
                    }
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        # Alias for client compatibility
        @api.post("/register_voice")
        async def register_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None)
        ):
            """Alias for /clone_voice for client compatibility."""
            return await clone_voice(voice_id=voice_id, audio_file=audio_file, audio_url=audio_url)

        @api.delete("/voices/{voice_id}")
        async def delete_voice(voice_id: str):
            """Delete a registered voice."""
            if voice_id in self._builtin_voices:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Cannot delete builtin voice '{voice_id}'"}
                )

            ref_path = f"/voice-cache/references/{voice_id}.wav"

            if not os.path.exists(ref_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found"}
                )

            try:
                os.unlink(ref_path)
                self._warmed_voices.discard(voice_id)
                self._voice_states.pop(voice_id, None)  # Clear cached state
                voice_cache.commit()
                return {"status": "deleted", "voice_id": voice_id}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        return api


@app.local_entrypoint()
def main():
    """Test the service locally."""
    print("Pocket TTS service ready")
    print("Deploy with: modal deploy modal/pocket_tts_service.py")
    print("\nFeatures:")
    print("  - 100M parameter model")
    print("  - Voice cloning from ~5 seconds of audio")
    print("  - ~200ms first-audio latency")
    print("  - Runs entirely on CPU")
