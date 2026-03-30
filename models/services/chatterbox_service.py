"""
Chatterbox Turbo TTS Service - Fast voice cloning.

Capabilities:
- Voice cloning from audio samples (no training required)
- Emotion exaggeration control
- Paralinguistic tags ([laugh], [cough], etc.)
- ~100-200ms streaming latency
- MIT licensed

Deploy: modal deploy modal/chatterbox_service.py
"""

import modal
import os

app = modal.App("phony-chatterbox-tts")

# Volume for storing cloned voice references
voice_cache = modal.Volume.from_name("phony-chatterbox-voices", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy<2",
        "scipy",
        "fastapi",
        "uvicorn",
        "soundfile",
        "requests",
    )
    .run_commands(
        "pip install git+https://github.com/resemble-ai/chatterbox.git"
    )
)


@app.cls(
    image=image,
    gpu="A10G",  # Chatterbox benefits from more VRAM
    timeout=600,
    container_idle_timeout=180,  # Keep warm for 3 minutes
    volumes={"/voice-cache": voice_cache},
    allow_concurrent_inputs=5,
)
class ChatterboxService:
    """Chatterbox TTS service with voice cloning."""

    @modal.enter()
    def load_model(self):
        """Load default Chatterbox model and set up lazy model cache."""
        import torch

        # Lazy model cache — keyed by model variant name
        self._models: dict[str, object] = {}

        # Create voice cache directory
        os.makedirs("/voice-cache/references", exist_ok=True)

        # In-memory cache for voice reference tensors (avoid disk reads)
        self._ref_cache: dict[str, torch.Tensor] = {}
        self._max_cache_size = 20  # Cache up to 20 voices

        # Pre-load turbo (default) so first request is fast
        self._get_model("turbo")
        print("Chatterbox turbo model loaded (other variants loaded on demand)")

    def _get_model(self, model_name: str):
        """Lazy-load a Chatterbox model variant."""
        if model_name in self._models:
            return self._models[model_name]

        if model_name in ("turbo", "original"):
            from chatterbox.tts import ChatterboxTTS
            self._models[model_name] = ChatterboxTTS.from_pretrained(device="cuda")
            print(f"Loaded Chatterbox {model_name} model")
        elif model_name == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self._models[model_name] = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            print(f"Loaded Chatterbox multilingual model")
        else:
            raise ValueError(f"Unknown model variant: {model_name}")

        return self._models[model_name]

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
                "service": "chatterbox-tts",
                "loaded_models": list(self._models.keys()),
                "available_models": ["turbo", "original", "multilingual"],
                "version": "v3",
                "features": ["voice_cloning", "emotion_control", "paralinguistic_tags", "multi_model", "multilingual"]
            }

        @api.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None)
        ):
            """Register a voice for cloning from audio sample.

            Either upload audio_file or provide audio_url.
            Recommended: 10-15 seconds of clear speech.
            """
            if not audio_file and not audio_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Either audio_file or audio_url required"}
                )

            try:
                # Get audio data
                if audio_url:
                    # Download from URL
                    response = requests.get(audio_url, timeout=30)
                    audio_data = response.content
                else:
                    audio_data = await audio_file.read()

                # Save to temporary file for processing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    f.write(audio_data)

                # Convert to proper format (16kHz mono WAV)
                import subprocess
                ref_path = f"/voice-cache/references/{voice_id}.wav"
                result = subprocess.run([
                    "ffmpeg", "-y", "-i", temp_path,
                    "-ar", "16000", "-ac", "1", "-f", "wav", ref_path
                ], capture_output=True)

                os.unlink(temp_path)

                if result.returncode != 0:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Failed to process audio", "details": result.stderr.decode()[:500]}
                    )

                # Verify the file was created
                if not os.path.exists(ref_path):
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Failed to save voice reference"}
                    )

                # Commit volume changes
                voice_cache.commit()

                return {
                    "status": "success",
                    "voice_id": voice_id,
                    "message": f"Voice '{voice_id}' registered for cloning"
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/warmup")
        async def warmup(request: Request):
            """Warm up a voice by pre-loading reference audio.

            Call this when a chat session starts to minimize first synthesis latency.
            """
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

            voice_id = body.get("voice_id")
            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            ref_path = f"/voice-cache/references/{voice_id}.wav"
            if not os.path.exists(ref_path):
                return JSONResponse(status_code=404, content={"error": f"Voice '{voice_id}' not found"})

            model_name = body.get("model", "turbo")

            # Warm up with a short synthesis to pre-load model caches
            start = time.time()
            try:
                model = self._get_model(model_name)
                _ = model.generate(
                    text="Hello.",
                    audio_prompt_path=ref_path,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
                elapsed = time.time() - start
                print(f"Warmed up voice '{voice_id}' in {elapsed:.2f}s")
                return {"status": "warmed", "voice_id": voice_id, "warmup_time": elapsed}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.get("/voices")
        def list_voices():
            """List registered voice references."""
            voices = []
            ref_dir = "/voice-cache/references"
            if os.path.exists(ref_dir):
                for f in os.listdir(ref_dir):
                    if f.endswith(".wav"):
                        voice_id = f[:-4]
                        voices.append({
                            "id": voice_id,
                            "type": "cloned"
                        })
            return {"voices": voices}

        def get_cached_ref_path(voice_id: str) -> str | None:
            """Get reference audio path, checking it exists."""
            ref_path = f"/voice-cache/references/{voice_id}.wav"
            return ref_path if os.path.exists(ref_path) else None

        @api.post("/synthesize")
        async def synthesize(request: Request):
            """Synthesize speech with voice cloning.

            Request body:
            {
                "text": "Text to synthesize",
                "voice_id": "huberman",  // ID of cloned voice
                "exaggeration": 0.5,  // 0.0-1.0, emotion intensity
                "cfg_weight": 0.5  // 0.0-1.0, prompt adherence
            }

            Paralinguistic tags supported in text:
            [laugh], [chuckle], [cough], [sigh], etc.

            Returns: WAV audio
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")
            exaggeration = body.get("exaggeration", 0.5)
            cfg_weight = body.get("cfg_weight", 0.5)
            model_name = body.get("model", "turbo")
            language_id = body.get("language_id")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            # Load voice reference
            ref_path = get_cached_ref_path(voice_id)
            if not ref_path:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found. Clone it first with /clone_voice"}
                )

            try:
                load_time = time.time()

                model = self._get_model(model_name)

                # Build generation kwargs
                gen_kwargs = {
                    "text": text,
                    "audio_prompt_path": ref_path,
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                }
                if model_name == "multilingual" and language_id:
                    gen_kwargs["language_id"] = language_id

                # Generate speech with reference audio path
                wav = model.generate(**gen_kwargs)

                gen_time = time.time() - load_time

                # Convert to numpy
                audio_np = wav.squeeze().cpu().numpy()

                # Write to WAV buffer
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, 24000, format='WAV')
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"Synthesized {len(text)} chars with voice '{voice_id}' in {elapsed:.3f}s (gen: {gen_time:.3f}s)")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Inference-Time": f"{gen_time:.3f}",
                        "X-Voice-Id": voice_id
                    }
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/synthesize_stream")
        async def synthesize_stream(request: Request):
            """Synthesize speech with sentence-level streaming.

            Splits text into sentences and synthesizes in parallel,
            streaming back audio as each sentence completes.

            Returns: Audio chunks as multipart stream for faster first-audio.
            """
            from fastapi.responses import StreamingResponse
            import re
            import asyncio
            import concurrent.futures

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")
            exaggeration = body.get("exaggeration", 0.5)
            cfg_weight = body.get("cfg_weight", 0.5)
            model_name = body.get("model", "turbo")
            language_id = body.get("language_id")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            ref_path = get_cached_ref_path(voice_id)
            if not ref_path:
                return JSONResponse(status_code=404, content={"error": f"Voice '{voice_id}' not found"})

            # Split into sentences (keep punctuation)
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return JSONResponse(status_code=400, content={"error": "No sentences found"})

            print(f"Streaming synthesis: {len(sentences)} sentences for voice '{voice_id}' (model={model_name})")

            # Capture model reference for nested function
            model = self._get_model(model_name)

            def synthesize_sentence(sentence: str) -> bytes:
                """Synthesize a single sentence."""
                gen_kwargs = {
                    "text": sentence,
                    "audio_prompt_path": ref_path,
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                }
                if model_name == "multilingual" and language_id:
                    gen_kwargs["language_id"] = language_id
                wav = model.generate(**gen_kwargs)
                audio_np = wav.squeeze().cpu().numpy()
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, 24000, format='WAV')
                buffer.seek(0)
                return buffer.read()

            async def stream_audio():
                """Stream audio chunks as they complete."""
                loop = asyncio.get_event_loop()

                # Process sentences - first one immediately, rest in parallel
                for i, sentence in enumerate(sentences):
                    start = time.time()
                    audio_data = await loop.run_in_executor(None, synthesize_sentence, sentence)
                    elapsed = time.time() - start
                    print(f"  Sentence {i+1}/{len(sentences)}: {len(sentence)} chars in {elapsed:.2f}s")
                    yield audio_data

            return StreamingResponse(
                stream_audio(),
                media_type="audio/wav",
                headers={"X-Sentences": str(len(sentences))}
            )

        @api.post("/synthesize_with_url")
        async def synthesize_with_url(request: Request):
            """One-shot voice cloning from URL (no pre-registration needed).

            Request body:
            {
                "text": "Text to synthesize",
                "speaker_wav_url": "https://...",  // URL to voice sample
                "exaggeration": 0.5,
                "cfg_weight": 0.5
            }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            speaker_wav_url = body.get("speaker_wav_url")
            exaggeration = body.get("exaggeration", 0.5)
            cfg_weight = body.get("cfg_weight", 0.5)
            model_name = body.get("model", "turbo")
            language_id = body.get("language_id")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            if not speaker_wav_url:
                return JSONResponse(status_code=400, content={"error": "speaker_wav_url is required"})

            try:
                # Download and convert voice sample
                import subprocess
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                result = subprocess.run([
                    "ffmpeg", "-y",
                    "-headers", "User-Agent: Mozilla/5.0",
                    "-i", speaker_wav_url,
                    "-ar", "16000", "-ac", "1", "-t", "30",  # Max 30 seconds
                    "-f", "wav", temp_path
                ], capture_output=True)

                if result.returncode != 0:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Failed to download/convert audio"}
                    )

                model = self._get_model(model_name)
                gen_kwargs = {
                    "text": text,
                    "audio_prompt_path": temp_path,
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                }
                if model_name == "multilingual" and language_id:
                    gen_kwargs["language_id"] = language_id

                wav = model.generate(**gen_kwargs)
                os.unlink(temp_path)

                # Convert to numpy
                audio_np = wav.squeeze().cpu().numpy()

                # Write to WAV buffer
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, 24000, format='WAV')
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"One-shot synthesis in {elapsed:.3f}s")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={"X-Generation-Time": f"{elapsed:.3f}"}
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        return api


@app.local_entrypoint()
def main():
    """Test the service locally."""
    print("Chatterbox TTS service ready")
    print("Deploy with: modal deploy modal/chatterbox_service.py")
