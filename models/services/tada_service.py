"""
TADA 1B TTS Service - Zero-hallucination speech generation by Hume AI.

Capabilities:
- Text-acoustic dual alignment (zero hallucinations in 1000+ tests)
- RTF 0.09 (5x faster than comparable LLM-based TTS)
- Voice cloning from reference audio + text
- Prompt caching for repeated voice reuse
- Multilingual (ar, ch, de, es, fr, it, ja, pl, pt)
- Based on Llama 3.2 1B

Deploy: modal deploy modal/tada_service.py
"""

import modal
import os

app = modal.App("phony-tada-tts")

voice_cache = modal.Volume.from_name("phony-tada-voices", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.7.0,<2.8.0",
        "torchaudio",
        "torchvision",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "requests",
        "transformers>=4.57.1,<5",
        "descript-audio-codec>=1.0.0",
        "hume-tada",
    )
    .run_commands(
        # Pre-download model weights into the image
        "python -c \""
        "from tada.modules.encoder import Encoder; "
        "from tada.modules.tada import TadaForCausalLM; "
        "import torch; "
        "Encoder.from_pretrained('HumeAI/tada-codec', subfolder='encoder'); "
        "TadaForCausalLM.from_pretrained('HumeAI/tada-1b', torch_dtype=torch.bfloat16); "
        "print('TADA models downloaded')"
        "\""
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
class TadaService:
    """TADA 1B TTS service with zero-hallucination speech generation."""

    @modal.enter()
    def load_model(self):
        """Load TADA encoder and model on startup."""
        import torch
        from tada.modules.encoder import Encoder
        from tada.modules.tada import TadaForCausalLM

        self.device = "cuda"
        self.encoder = Encoder.from_pretrained(
            "HumeAI/tada-codec", subfolder="encoder"
        ).to(self.device)
        self.model = TadaForCausalLM.from_pretrained(
            "HumeAI/tada-1b", torch_dtype=torch.bfloat16
        ).to(self.device)

        print("TADA 1B encoder + model loaded")

        os.makedirs("/voice-cache/references", exist_ok=True)
        os.makedirs("/voice-cache/prompts", exist_ok=True)

        # In-memory prompt cache for fast repeated synthesis
        self._prompt_cache = {}

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with TTS endpoints."""
        from fastapi import FastAPI, Request, UploadFile, File, Form
        from fastapi.responses import Response, JSONResponse
        import soundfile as sf
        import torchaudio
        import torch
        import io
        import time
        import tempfile
        import subprocess
        from tada.modules.encoder import EncoderOutput

        api = FastAPI()

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "tada-tts",
                "model": "TADA-1B",
                "features": [
                    "zero_hallucination",
                    "voice_cloning",
                    "prompt_caching",
                    "multilingual",
                ],
                "rtf": 0.09,
            }

        @api.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            reference_text: str = Form(""),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            """Register a voice reference for cloning.

            TADA uses reference audio + optional reference text for encoding.
            The encoded prompt is cached for fast repeated synthesis.

            Recommended: 5-15 seconds of clear speech with matching transcript.
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

                # Normalize to mono WAV at model sample rate
                ref_path = f"/voice-cache/references/{voice_id}.wav"
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", temp_path,
                        "-ar", "16000", "-ac", "1", "-f", "wav", ref_path,
                    ],
                    capture_output=True,
                )
                os.unlink(temp_path)

                if result.returncode != 0:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Failed to process audio"},
                    )

                # Pre-encode the prompt for fast synthesis
                audio, sample_rate = torchaudio.load(ref_path)
                text_list = [reference_text] if reference_text else None
                prompt = self.encoder(audio, text=text_list, sample_rate=sample_rate)

                # Save to disk and in-memory cache
                prompt_path = f"/voice-cache/prompts/{voice_id}.pt"
                prompt.save(prompt_path)
                self._prompt_cache[voice_id] = prompt

                voice_cache.commit()

                return {
                    "status": "success",
                    "voice_id": voice_id,
                    "prompt_cached": True,
                    "message": f"Voice '{voice_id}' registered and prompt encoded",
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/register_voice")
        async def register_voice(
            voice_id: str = Form(...),
            reference_text: str = Form(""),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            """Alias for /clone_voice (matches platform API)."""
            return await clone_voice(
                voice_id=voice_id,
                reference_text=reference_text,
                audio_file=audio_file,
                audio_url=audio_url,
            )

        @api.get("/voices")
        def list_voices():
            """List registered voice references."""
            voices = []
            ref_dir = "/voice-cache/references"
            if os.path.exists(ref_dir):
                for f in os.listdir(ref_dir):
                    if f.endswith(".wav"):
                        vid = f[:-4]
                        voices.append({
                            "id": vid,
                            "type": "cloned",
                            "prompt_cached": vid in self._prompt_cache
                            or os.path.exists(f"/voice-cache/prompts/{vid}.pt"),
                        })
            return {"voices": voices}

        @api.delete("/voices/{voice_id}")
        def delete_voice(voice_id: str):
            """Delete a cloned voice."""
            ref_path = f"/voice-cache/references/{voice_id}.wav"
            prompt_path = f"/voice-cache/prompts/{voice_id}.pt"
            if not os.path.exists(ref_path):
                return JSONResponse(status_code=404, content={"error": "Voice not found"})

            os.unlink(ref_path)
            if os.path.exists(prompt_path):
                os.unlink(prompt_path)
            self._prompt_cache.pop(voice_id, None)
            voice_cache.commit()
            return {"status": "deleted", "voice_id": voice_id}

        # Capture service instance for use in route closures
        _svc = self

        def _get_prompt(voice_id: str):
            """Load a cached prompt from memory or disk."""
            if voice_id in _svc._prompt_cache:
                return _svc._prompt_cache[voice_id]

            prompt_path = f"/voice-cache/prompts/{voice_id}.pt"
            if os.path.exists(prompt_path):
                prompt = EncoderOutput.load(prompt_path, device=_svc.device)
                _svc._prompt_cache[voice_id] = prompt
                return prompt

            # Fall back to re-encoding from reference audio
            ref_path = f"/voice-cache/references/{voice_id}.wav"
            if os.path.exists(ref_path):
                audio, sample_rate = torchaudio.load(ref_path)
                prompt = _svc.encoder(audio, sample_rate=sample_rate)
                _svc._prompt_cache[voice_id] = prompt
                return prompt

            return None

        @api.post("/synthesize")
        async def synthesize(request: Request):
            """Synthesize speech with voice cloning.

            Request body:
            {
                "text": "Text to synthesize",
                "voice_id": "huberman"
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

            prompt = _get_prompt(voice_id)
            if prompt is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found. Register it first with /clone_voice"},
                )

            try:
                output = _svc.model.generate(prompt=prompt, text=text)

                # output is a named tuple / object with .audio and .sample_rate
                audio_tensor = output.audio
                sr = output.sample_rate

                # Write WAV
                buffer = io.BytesIO()
                if hasattr(audio_tensor, "cpu"):
                    audio_np = audio_tensor.cpu().numpy()
                else:
                    audio_np = audio_tensor
                sf.write(buffer, audio_np, sr, format="WAV")
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"TADA synthesized {len(text)} chars with voice '{voice_id}' in {elapsed:.3f}s")

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

        @api.post("/synthesize_with_url")
        async def synthesize_with_url(request: Request):
            """One-shot cloning: synthesize using reference audio URL (no pre-registration).

            Request body:
            {
                "text": "Text to synthesize",
                "speaker_wav_url": "https://...",
                "reference_text": ""  // optional transcript of reference audio
            }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            speaker_wav_url = body.get("speaker_wav_url")
            reference_text = body.get("reference_text", "")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not speaker_wav_url:
                return JSONResponse(status_code=400, content={"error": "speaker_wav_url is required"})

            try:
                # Download and normalize reference audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                import requests as req
                raw = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
                raw.write(req.get(speaker_wav_url, timeout=30).content)
                raw.close()

                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", "-t", "30", "-f", "wav", temp_path],
                    capture_output=True,
                )
                os.unlink(raw.name)

                # Encode prompt on the fly
                audio, sample_rate = torchaudio.load(temp_path)
                text_list = [reference_text] if reference_text else None
                prompt = _svc.encoder(audio, text=text_list, sample_rate=sample_rate)
                os.unlink(temp_path)

                # Generate
                output = _svc.model.generate(prompt=prompt, text=text)
                audio_tensor = output.audio
                sr = output.sample_rate

                buffer = io.BytesIO()
                if hasattr(audio_tensor, "cpu"):
                    audio_np = audio_tensor.cpu().numpy()
                else:
                    audio_np = audio_tensor
                sf.write(buffer, audio_np, sr, format="WAV")
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"TADA one-shot synthesis in {elapsed:.3f}s")

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
    print("TADA 1B TTS service ready")
    print("Deploy with: modal deploy modal/tada_service.py")
