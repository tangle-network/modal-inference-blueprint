"""
Higgs Audio V2 TTS Service - 3B-param text-audio foundation model by Boson AI.

Capabilities:
- Multi-speaker dialogue generation
- Melodic humming and singing
- Zero-shot voice cloning from reference audio
- Smart voice (no reference needed, style description)
- Beats gpt-4o-mini-tts on benchmarks
- 10M+ hours pretraining data

Deploy: modal deploy modal/higgs_audio_service.py
"""

import modal
import os

app = modal.App("phony-higgs-audio-tts")

voice_cache = modal.Volume.from_name("phony-higgs-audio-voices", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.02-py3",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "git")
    .pip_install(
        "soundfile",
        "fastapi",
        "uvicorn",
        "requests",
        "descript-audio-codec",
        "transformers>=4.45.1,<4.47.0",
        "librosa",
        "dacite",
        "torchaudio",
        "json_repair",
        "pydantic",
        "vector_quantize_pytorch",
        "loguru",
        "pydub",
        "omegaconf",
        "click",
        "langid",
        "jieba",
        "accelerate>=0.26.0",
    )
    .run_commands(
        "pip install git+https://github.com/boson-ai/higgs-audio.git",
        # Pre-download model weights
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('bosonai/higgs-audio-v2-generation-3B-base'); "
        "snapshot_download('bosonai/higgs-audio-v2-tokenizer'); "
        "print('Higgs Audio V2 models downloaded')"
        "\"",
    )
)


@app.cls(
    image=image,
    gpu="A100",  # 3B params requires >= 24GB VRAM
    timeout=600,
    container_idle_timeout=120,
    volumes={"/voice-cache": voice_cache},
    allow_concurrent_inputs=3,  # Lower concurrency due to model size
)
class HiggsAudioService:
    """Higgs Audio V2 TTS service with multi-speaker and voice cloning."""

    MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
    TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

    @modal.enter()
    def load_model(self):
        """Load Higgs Audio V2 model on startup."""
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

        self.engine = HiggsAudioServeEngine(
            self.MODEL_PATH,
            self.TOKENIZER_PATH,
            device="cuda",
        )
        print("Higgs Audio V2 (3B) loaded on CUDA")

        os.makedirs("/voice-cache/references", exist_ok=True)

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with TTS endpoints."""
        from fastapi import FastAPI, Request, UploadFile, File, Form
        from fastapi.responses import Response, JSONResponse
        from boson_multimodal.data_types import ChatMLSample, Message
        import soundfile as sf
        import torch
        import torchaudio
        import io
        import time
        import tempfile
        import subprocess
        import numpy as np

        api = FastAPI()

        AUDIO_PLACEHOLDER = "<|audio|>"

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "service": "higgs-audio-v2",
                "model": "higgs-audio-v2-generation-3B-base",
                "features": [
                    "multi_speaker",
                    "voice_cloning",
                    "smart_voice",
                    "melodic_humming",
                    "emotion_control",
                ],
                "gpu": "A100",
            }

        @api.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            """Register a voice reference for cloning.

            Higgs Audio uses reference audio in the system message.
            Recommended: 5-15 seconds of clear speech.
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
                    "message": f"Voice '{voice_id}' registered for cloning",
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
            """Alias for /clone_voice (matches platform API)."""
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

        # Keep a ref to self for use inside route closures
        _engine = self.engine

        def _load_ref_audio(voice_id: str):
            """Load reference audio as tensor for system message injection."""
            ref_path = f"/voice-cache/references/{voice_id}.wav"
            if not os.path.exists(ref_path):
                return None, None
            audio, sr = torchaudio.load(ref_path)
            return audio, sr

        @api.post("/synthesize")
        async def synthesize(request: Request):
            """Synthesize speech with optional voice cloning and emotion.

            Request body:
            {
                "text": "Text to synthesize",
                "voice_id": "huberman",         // optional — use cloned voice
                "emotion": "cheerful",           // optional — style description
                "scene": "podcast interview",    // optional — scene/context description
                "temperature": 0.3,              // optional, default 0.3
                "max_new_tokens": 2048           // optional, default 2048
            }

            Without voice_id, uses Higgs Audio's "smart voice" (model picks voice).
            Returns: WAV audio (24kHz mono)
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")
            emotion = body.get("emotion")
            scene = body.get("scene")
            temperature = body.get("temperature", 0.3)
            max_new_tokens = body.get("max_new_tokens", 2048)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            try:
                # Build system message
                system_parts = ["Generate audio following instruction."]
                if emotion:
                    system_parts.append(f"Speak with a {emotion} tone.")
                if scene:
                    system_parts.append(f"Context: {scene}.")

                audio_ids = []

                # If voice_id is provided, inject reference audio into system message
                if voice_id:
                    ref_path = f"/voice-cache/references/{voice_id}.wav"
                    if not os.path.exists(ref_path):
                        return JSONResponse(
                            status_code=404,
                            content={"error": f"Voice '{voice_id}' not found. Register with /clone_voice"},
                        )
                    system_parts.append(
                        f"Use the following voice as reference: {AUDIO_PLACEHOLDER}"
                    )
                    audio_ids.append(ref_path)

                system_content = " ".join(system_parts)

                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=text),
                ]

                sample = ChatMLSample(messages=messages)

                # If we have audio references, attach them
                if audio_ids:
                    sample.audio_ids = audio_ids

                output = _engine.generate(
                    chat_ml_sample=sample,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                )

                # output has .audio (numpy array) and .sampling_rate
                audio_np = output.audio
                sr = output.sampling_rate

                buffer = io.BytesIO()
                sf.write(buffer, audio_np, sr, format="WAV")
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(
                    f"Higgs Audio synthesized {len(text)} chars"
                    f" (voice={voice_id or 'smart'}, emotion={emotion or 'none'})"
                    f" in {elapsed:.3f}s"
                )

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Voice-Id": voice_id or "smart",
                        "X-Emotion": emotion or "none",
                        "X-Sample-Rate": str(sr),
                    },
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/synthesize_dialogue")
        async def synthesize_dialogue(request: Request):
            """Generate multi-speaker dialogue audio.

            Request body:
            {
                "transcript": "Speaker 1: Hello!\\nSpeaker 2: Hey there!",
                "voice_ids": ["alice", "bob"],   // optional — one per speaker
                "temperature": 0.3,
                "max_new_tokens": 4096
            }

            Returns: WAV audio with multi-speaker dialogue
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            transcript = body.get("transcript", "")
            voice_ids = body.get("voice_ids", [])
            temperature = body.get("temperature", 0.3)
            max_new_tokens = body.get("max_new_tokens", 4096)

            if not transcript:
                return JSONResponse(status_code=400, content={"error": "transcript is required"})

            try:
                system_parts = ["Generate a multi-speaker dialogue following the transcript."]
                audio_ids = []

                # Inject voice references if provided
                for i, vid in enumerate(voice_ids):
                    ref_path = f"/voice-cache/references/{vid}.wav"
                    if os.path.exists(ref_path):
                        system_parts.append(
                            f"Speaker {i + 1} voice reference: {AUDIO_PLACEHOLDER}"
                        )
                        audio_ids.append(ref_path)

                system_content = " ".join(system_parts)

                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=transcript),
                ]

                sample = ChatMLSample(messages=messages)
                if audio_ids:
                    sample.audio_ids = audio_ids

                output = _engine.generate(
                    chat_ml_sample=sample,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                )

                audio_np = output.audio
                sr = output.sampling_rate

                buffer = io.BytesIO()
                sf.write(buffer, audio_np, sr, format="WAV")
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"Higgs Audio dialogue synthesis in {elapsed:.3f}s")

                return Response(
                    content=buffer.read(),
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Speakers": str(len(voice_ids)),
                        "X-Sample-Rate": str(sr),
                    },
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/synthesize_with_url")
        async def synthesize_with_url(request: Request):
            """One-shot cloning: synthesize using reference audio URL (no pre-registration)."""
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            speaker_wav_url = body.get("speaker_wav_url")
            emotion = body.get("emotion")
            temperature = body.get("temperature", 0.3)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not speaker_wav_url:
                return JSONResponse(status_code=400, content={"error": "speaker_wav_url is required"})

            try:
                # Download and normalize
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                import requests as req
                raw = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
                raw.write(req.get(speaker_wav_url, timeout=30).content)
                raw.close()

                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw.name, "-ar", "24000", "-ac", "1", "-t", "30", "-f", "wav", temp_path],
                    capture_output=True,
                )
                os.unlink(raw.name)

                system_parts = [
                    "Generate audio following instruction.",
                    f"Use the following voice as reference: {AUDIO_PLACEHOLDER}",
                ]
                if emotion:
                    system_parts.append(f"Speak with a {emotion} tone.")

                messages = [
                    Message(role="system", content=" ".join(system_parts)),
                    Message(role="user", content=text),
                ]

                sample = ChatMLSample(messages=messages)
                sample.audio_ids = [temp_path]

                output = _engine.generate(
                    chat_ml_sample=sample,
                    max_new_tokens=2048,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                )

                os.unlink(temp_path)

                audio_np = output.audio
                sr = output.sampling_rate

                buffer = io.BytesIO()
                sf.write(buffer, audio_np, sr, format="WAV")
                buffer.seek(0)

                elapsed = time.time() - start_time
                print(f"Higgs Audio one-shot synthesis in {elapsed:.3f}s")

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
    print("Higgs Audio V2 service ready")
    print("Deploy with: modal deploy modal/higgs_audio_service.py")
