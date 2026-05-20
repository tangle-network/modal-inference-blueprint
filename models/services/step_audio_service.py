"""
Step-Audio 2 Mini S2S Service - StepFun's 8B speech-to-speech model.

Capabilities:
- End-to-end speech-to-speech conversation (surpasses GPT-4o-Audio)
- ASR, translation, audio understanding, paralinguistic analysis
- Multi-turn conversation with audio context
- Tool calling and web search integration
- Voice cloning via prompt WAV (CosyVoice2 vocoder)
- 8B parameters, 24kHz output audio
- Apache-2.0 license

Architecture:
- StepAudio2 LLM: audio input -> interleaved text + audio tokens
- Token2wav vocoder: audio tokens -> 24kHz WAV via CosyVoice2 flow + HiFi-GAN
- Speaker embedding via CamPlus for voice cloning

Endpoints:
- GET  /health     - Service health check
- POST /converse   - Single-turn S2S (audio in -> audio + text out)
- WS   /stream     - Streaming S2S conversation with multi-turn context
- POST /asr        - Speech-to-text only
- POST /tts        - Text-to-speech only

Deploy: modal deploy modal/step_audio_service.py
"""

import modal
import os

app = modal.App("modal-inference-step-audio-s2s")

MODEL_DIR = "/models/step-audio-2"
model_cache = modal.Volume.from_name("phony-step-audio-model-cache", create_if_missing=True)
voice_cache = modal.Volume.from_name("phony-step-audio-voices", create_if_missing=True)

SAMPLE_RATE = 24000


def download_models():
    """Download Step-Audio-2-mini at image build time."""
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Downloading stepfun-ai/Step-Audio-2-mini...")
    snapshot_download(
        repo_id="stepfun-ai/Step-Audio-2-mini",
        local_dir=f"{MODEL_DIR}/Step-Audio-2-mini",
    )
    print("Step-Audio-2-mini downloaded")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch>=2.3",
        "torchaudio",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "transformers==4.49.0",
        "librosa",
        "onnxruntime",
        "s3tokenizer",
        "diffusers",
        "hyperpyyaml",
        "huggingface_hub",
        "hf_transfer",
        "accelerate",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .run_commands(
        "pip install git+https://github.com/stepfun-ai/Step-Audio2.git || true",
    )
    .run_function(download_models)
)


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    container_idle_timeout=300,
    volumes={
        MODEL_DIR: model_cache,
        "/voice-cache": voice_cache,
    },
    allow_concurrent_inputs=2,
)
class StepAudioService:
    """Step-Audio 2 Mini S2S service for speech conversation."""

    @modal.enter()
    def setup(self):
        import sys
        import torch

        # Step-Audio2 repo files are installed or we load them from the model dir
        model_path = f"{MODEL_DIR}/Step-Audio-2-mini"

        # Add repo to path if installed from git
        if os.path.exists(f"{model_path}/stepaudio2.py"):
            sys.path.insert(0, model_path)

        # Import model classes -- try installed package first, fall back to local
        try:
            from stepaudio2 import StepAudio2
            from token2wav import Token2wav
        except ImportError:
            # Load from model directory
            import importlib.util
            for mod_name, mod_file in [
                ("utils", "utils.py"),
                ("stepaudio2", "stepaudio2.py"),
                ("token2wav", "token2wav.py"),
            ]:
                spec = importlib.util.spec_from_file_location(
                    mod_name, f"{model_path}/{mod_file}",
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = mod
                    spec.loader.exec_module(mod)

            from stepaudio2 import StepAudio2
            from token2wav import Token2wav

        self.model = StepAudio2(model_path)
        self.token2wav = Token2wav(f"{model_path}/token2wav")

        # Default prompt WAV paths
        self._default_female = f"{model_path}/assets/default_female.wav"
        self._default_male = f"{model_path}/assets/default_male.wav"

        print("Step-Audio 2 Mini loaded on CUDA")

    def _get_prompt_wav(self, voice: str | None) -> str:
        """Resolve voice to a prompt WAV path."""
        if voice == "male":
            return self._default_male
        if voice and os.path.exists(f"/voice-cache/references/{voice}.wav"):
            return f"/voice-cache/references/{voice}.wav"
        return self._default_female

    def _do_s2s(
        self,
        audio_path: str,
        system_prompt: str = "You are a helpful assistant.",
        voice: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> tuple:
        """Run speech-to-speech: audio in -> (text, audio_bytes)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": "<tts_start>", "eot": False},
        ]

        tokens, text, audio_tokens = self.model(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.05,
            top_p=0.9,
        )

        # Remove audio padding tokens
        audio_tokens = [x for x in audio_tokens if x < 6561]
        prompt_wav = self._get_prompt_wav(voice)
        audio_bytes = self.token2wav(audio_tokens, prompt_wav=prompt_wav)

        return text, audio_bytes, tokens

    def _do_asr(self, audio_path: str) -> str:
        """Speech-to-text only."""
        messages = [
            {"role": "system", "content": "Please transcribe the speech accurately."},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
            {"role": "assistant", "content": None},
        ]
        _, text, _ = self.model(messages, max_new_tokens=256)
        return text

    def _do_tts(
        self,
        text: str,
        voice: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> bytes:
        """Text-to-speech only."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "human", "content": text},
            {"role": "assistant", "content": "<tts_start>", "eot": False},
        ]

        tokens, _, audio_tokens = self.model(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.05,
            top_p=0.9,
        )

        audio_tokens = [x for x in audio_tokens if x < 6561]
        prompt_wav = self._get_prompt_wav(voice)
        return self.token2wav(audio_tokens, prompt_wav=prompt_wav)

    @modal.asgi_app()
    def web(self):
        import time
        import tempfile
        import subprocess
        from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse, Response, StreamingResponse

        web_app = FastAPI()
        svc = self

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "step-audio-2-mini",
                "gpu": "A100",
                "sample_rate": SAMPLE_RATE,
                "features": [
                    "speech_to_speech", "asr", "tts",
                    "multi_turn", "tool_calling", "translation",
                    "voice_cloning", "paralinguistic",
                ],
                "params": "8B",
            }

        @web_app.post("/converse")
        async def converse(request: Request):
            """Single-turn S2S: audio input -> audio + text response.

            Accepts multipart form with audio_file or JSON with audio_url.
            Returns audio/wav with X-Transcript header.
            """
            start = time.time()
            temp_path = None

            try:
                content_type = request.headers.get("content-type", "")

                if "multipart" in content_type:
                    form = await request.form()
                    audio_file = form.get("audio_file")
                    voice = form.get("voice", "female")
                    system_prompt = form.get("system_prompt", "You are a helpful assistant.")

                    if audio_file is None:
                        return JSONResponse(status_code=400, content={"error": "audio_file required"})

                    audio_data = await audio_file.read()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name
                        f.write(audio_data)

                    # Convert to 16kHz mono WAV
                    converted = temp_path + ".converted.wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", temp_path, "-ar", "16000", "-ac", "1", "-f", "wav", converted],
                        capture_output=True, check=True,
                    )
                    os.unlink(temp_path)
                    temp_path = converted

                else:
                    body = await request.json()
                    audio_url = body.get("audio_url")
                    voice = body.get("voice", "female")
                    system_prompt = body.get("system_prompt", "You are a helpful assistant.")

                    if not audio_url:
                        return JSONResponse(status_code=400, content={"error": "audio_url or audio_file required"})

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name

                    subprocess.run(
                        ["ffmpeg", "-y", "-i", audio_url, "-ar", "16000", "-ac", "1", "-f", "wav", temp_path],
                        capture_output=True, check=True,
                    )

                text, audio_bytes, _ = svc._do_s2s(
                    temp_path,
                    system_prompt=system_prompt,
                    voice=voice,
                )
                elapsed = time.time() - start

                print(f"step-audio-s2s: converse in {elapsed:.3f}s, transcript={text[:80]}")

                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Transcript": text[:500],
                        "X-Sample-Rate": str(SAMPLE_RATE),
                    },
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        @web_app.post("/asr")
        async def asr(request: Request):
            """Speech-to-text transcription.

            Accepts multipart form with audio_file or JSON with audio_url.
            Returns JSON {"text": "..."}.
            """
            temp_path = None
            try:
                content_type = request.headers.get("content-type", "")

                if "multipart" in content_type:
                    form = await request.form()
                    audio_file = form.get("audio_file")
                    if audio_file is None:
                        return JSONResponse(status_code=400, content={"error": "audio_file required"})

                    audio_data = await audio_file.read()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name
                        f.write(audio_data)

                    converted = temp_path + ".converted.wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", temp_path, "-ar", "16000", "-ac", "1", "-f", "wav", converted],
                        capture_output=True, check=True,
                    )
                    os.unlink(temp_path)
                    temp_path = converted
                else:
                    body = await request.json()
                    audio_url = body.get("audio_url")
                    if not audio_url:
                        return JSONResponse(status_code=400, content={"error": "audio_url or audio_file required"})

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", audio_url, "-ar", "16000", "-ac", "1", "-f", "wav", temp_path],
                        capture_output=True, check=True,
                    )

                text = svc._do_asr(temp_path)
                return {"text": text}

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        @web_app.post("/tts")
        async def tts(request: Request):
            """Text-to-speech synthesis.

            Body: {"text": "...", "voice": "female"|"male"|voice_id}
            Returns: audio/wav
            """
            start = time.time()
            try:
                body = await request.json()
                text = body.get("text", "")
                voice = body.get("voice", "female")

                if not text:
                    return JSONResponse(status_code=400, content={"error": "text is required"})

                audio_bytes = svc._do_tts(text, voice=voice)
                elapsed = time.time() - start

                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Sample-Rate": str(SAMPLE_RATE),
                    },
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @web_app.websocket("/stream")
        async def stream_converse(ws: WebSocket):
            """Multi-turn streaming S2S conversation.

            Protocol:
            - Client sends JSON: {"audio_url": "...", "voice": "female", "system_prompt": "..."}
              or binary audio data (WAV/PCM 16kHz mono)
            - Server sends JSON: {"type": "text", "content": "..."} for transcript
              then binary WAV audio for speech response
            - Send JSON: {"type": "end"} to close
            """
            import json

            await ws.accept()
            history = [{"role": "system", "content": "You are a helpful assistant."}]
            voice = "female"
            temp_path = None

            try:
                while True:
                    msg = await ws.receive()

                    if "text" in msg:
                        data = json.loads(msg["text"])

                        if data.get("type") == "end":
                            break

                        if "system_prompt" in data:
                            history = [{"role": "system", "content": data["system_prompt"]}]

                        if "voice" in data:
                            voice = data["voice"]

                        if "text" in data:
                            # Text input turn
                            history.append({"role": "human", "content": data["text"]})
                            history.append({"role": "assistant", "content": "<tts_start>", "eot": False})

                            tokens, text, audio_tokens = svc.model(
                                history,
                                max_new_tokens=4096,
                                temperature=0.7,
                                do_sample=True,
                                repetition_penalty=1.05,
                                top_p=0.9,
                            )
                            audio_tokens_clean = [x for x in audio_tokens if x < 6561]
                            prompt_wav = svc._get_prompt_wav(voice)
                            audio_bytes = svc.token2wav(audio_tokens_clean, prompt_wav=prompt_wav)

                            await ws.send_json({"type": "text", "content": text})
                            await ws.send_bytes(audio_bytes)

                            # Update history for multi-turn
                            history.pop()
                            history.append({
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": "<tts_start>"},
                                    {"type": "token", "token": tokens},
                                ],
                            })

                    elif "bytes" in msg:
                        # Binary audio input
                        audio_data = msg["bytes"]
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            temp_path = f.name
                            f.write(audio_data)

                        converted = temp_path + ".converted.wav"
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", temp_path, "-ar", "16000", "-ac", "1", "-f", "wav", converted],
                            capture_output=True, check=True,
                        )
                        os.unlink(temp_path)
                        temp_path = converted

                        history.append({
                            "role": "human",
                            "content": [{"type": "audio", "audio": temp_path}],
                        })
                        history.append({"role": "assistant", "content": "<tts_start>", "eot": False})

                        tokens, text, audio_tokens = svc.model(
                            history,
                            max_new_tokens=4096,
                            temperature=0.7,
                            do_sample=True,
                            repetition_penalty=1.05,
                            top_p=0.9,
                        )
                        audio_tokens_clean = [x for x in audio_tokens if x < 6561]
                        prompt_wav = svc._get_prompt_wav(voice)
                        audio_bytes = svc.token2wav(audio_tokens_clean, prompt_wav=prompt_wav)

                        await ws.send_json({"type": "text", "content": text})
                        await ws.send_bytes(audio_bytes)

                        history.pop()
                        history.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "<tts_start>"},
                                {"type": "token", "token": tokens},
                            ],
                        })

                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)
                            temp_path = None

            except WebSocketDisconnect:
                print("Step-Audio stream disconnected")
            except Exception as e:
                print(f"Step-Audio stream error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                await ws.close()

        @web_app.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            """Register a voice prompt WAV for S2S voice cloning."""
            if not audio_file and not audio_url:
                return JSONResponse(status_code=400, content={"error": "audio_file or audio_url required"})

            ref_path = f"/voice-cache/references/{voice_id}.wav"
            os.makedirs("/voice-cache/references", exist_ok=True)

            try:
                if audio_file:
                    data = await audio_file.read()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(data)
                        src = f.name
                else:
                    src = audio_url

                subprocess.run(
                    ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-t", "30", "-f", "wav", ref_path],
                    capture_output=True, check=True,
                )

                if audio_file and os.path.exists(src):
                    os.unlink(src)

                return {"status": "success", "voice_id": voice_id}

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        return web_app


@app.local_entrypoint()
def main():
    print("Step-Audio 2 Mini S2S service ready (8B, 24kHz)")
    print("Deploy with: modal deploy modal/step_audio_service.py")
