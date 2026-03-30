"""
IndexTTS-2 Service - Zero-shot voice cloning with emotion control.

Capabilities:
- Zero-shot voice cloning from reference audio
- Emotion decoupled from speaker (8-dim emotion vector)
- Precise duration control
- FP16 inference support
- ~200-400ms TTFB on A10G

Deploy: modal deploy modal/indextts_service.py
"""

import modal
import os

app = modal.App("phony-indextts2-tts")

voice_cache = modal.Volume.from_name("phony-indextts2-voices", create_if_missing=True)
model_cache = modal.Volume.from_name("phony-indextts2-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "espeak-ng", "build-essential")
    .pip_install(
        "torch==2.8.*",
        "torchaudio==2.8.*",
        "numpy==1.26.2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "requests",
        "transformers==4.52.1",
        "accelerate==1.8.1",
        "omegaconf>=2.3.0",
        "einops>=0.8.1",
        "librosa==0.10.2.post1",
        "safetensors==0.5.2",
        "sentencepiece>=0.2.1",
        "tokenizers==0.21.0",
        "g2p-en==2.1.0",
        "jieba==0.42.1",
        "munch==4.0.0",
        "descript-audiotools==0.7.2",
        "WeTextProcessing",
        "huggingface_hub[cli,hf_xet]",
    )
    .run_commands(
        "pip install git+https://github.com/index-tts/index-tts.git",
        # Download model weights into the image layer for fast cold starts
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='/model-weights')\"",
    )
)

# Emotion vector labels (order matters)
EMOTION_LABELS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=120,
    volumes={
        "/voice-cache": voice_cache,
        "/model-cache": model_cache,
    },
    allow_concurrent_inputs=5,
)
class IndexTTSService:
    """IndexTTS-2 service with zero-shot voice cloning and emotion control."""

    @modal.enter()
    def load_model(self):
        """Load IndexTTS-2 model on startup."""
        from indextts.infer_v2 import IndexTTS2

        model_dir = "/model-weights"
        cfg_path = os.path.join(model_dir, "config.yaml")

        self.model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        print("IndexTTS-2 model loaded (FP16)")

        os.makedirs("/voice-cache/references", exist_ok=True)

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with voice cloning TTS endpoints."""
        from fastapi import FastAPI, Request, UploadFile, File, Form
        from fastapi.responses import Response, JSONResponse
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
                "service": "indextts2",
                "model": "IndexTTS-2",
                "features": [
                    "zero_shot_cloning",
                    "emotion_vector",
                    "emotion_audio_ref",
                    "emotion_text",
                    "duration_control",
                ],
                "emotion_labels": EMOTION_LABELS,
            }

        @api.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            """Register a voice reference for zero-shot cloning.

            IndexTTS-2 uses reference audio at synthesis time.
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

        @api.post("/synthesize")
        async def synthesize(request: Request):
            """Synthesize speech with zero-shot voice cloning and optional emotion.

            Request body:
            {
                "text": "Text to synthesize",
                "voice_id": "huberman",
                "reference_audio_url": "https://...",  // alternative to voice_id
                "emotion": "happy",                    // named emotion (optional)
                "emotion_vector": [0,0,0.8,0,0,0,0,0], // 8-dim vector (optional)
                "emotion_audio_url": "https://..."     // emotion reference audio (optional)
            }

            Emotion vector order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

            Returns: WAV audio (24kHz mono)
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id")
            reference_audio_url = body.get("reference_audio_url")
            emotion = body.get("emotion")
            emotion_vector = body.get("emotion_vector")
            emotion_audio_url = body.get("emotion_audio_url")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})
            if not voice_id and not reference_audio_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "voice_id or reference_audio_url is required"},
                )

            # Resolve speaker audio path
            ref_path = None
            temp_ref = None
            if voice_id:
                ref_path = f"/voice-cache/references/{voice_id}.wav"
                if not os.path.exists(ref_path):
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"Voice '{voice_id}' not found. Register it first with /clone_voice"},
                    )
            elif reference_audio_url:
                import requests as req
                temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                raw = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
                raw.write(req.get(reference_audio_url, timeout=30).content)
                raw.close()
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw.name, "-ar", "24000", "-ac", "1", "-f", "wav", temp_ref.name],
                    capture_output=True,
                )
                os.unlink(raw.name)
                ref_path = temp_ref.name

            # Resolve emotion audio reference (if provided)
            emo_ref_path = None
            temp_emo = None
            if emotion_audio_url:
                import requests as req
                temp_emo = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                raw = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
                raw.write(req.get(emotion_audio_url, timeout=30).content)
                raw.close()
                subprocess.run(
                    ["ffmpeg", "-y", "-i", raw.name, "-ar", "24000", "-ac", "1", "-f", "wav", temp_emo.name],
                    capture_output=True,
                )
                os.unlink(raw.name)
                emo_ref_path = temp_emo.name

            try:
                # Build infer kwargs
                infer_kwargs = {
                    "spk_audio_prompt": ref_path,
                    "text": text,
                    "output_path": None,
                    "verbose": False,
                }

                # Emotion: vector > named > audio reference > text-based
                if emotion_vector:
                    if len(emotion_vector) != 8:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"emotion_vector must have 8 elements, got {len(emotion_vector)}"},
                        )
                    infer_kwargs["emo_vector"] = emotion_vector
                    infer_kwargs["use_random"] = False
                elif emotion and emotion in EMOTION_LABELS:
                    vec = [0.0] * 8
                    vec[EMOTION_LABELS.index(emotion)] = 0.8
                    infer_kwargs["emo_vector"] = vec
                    infer_kwargs["use_random"] = False
                elif emo_ref_path:
                    infer_kwargs["emo_audio_prompt"] = emo_ref_path

                # Generate to a temp file (IndexTTS-2 writes to file)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
                    out_path = out_f.name

                infer_kwargs["output_path"] = out_path
                self.model.infer(**infer_kwargs)

                # Read generated audio
                with open(out_path, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(out_path)

                elapsed = time.time() - start_time
                print(f"IndexTTS-2 synthesized {len(text)} chars in {elapsed:.3f}s")

                return Response(
                    content=audio_bytes,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Voice-Id": voice_id or "url-ref",
                        "X-Emotion": emotion or "none",
                    },
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_ref:
                    try:
                        os.unlink(temp_ref.name)
                    except OSError:
                        pass
                if temp_emo:
                    try:
                        os.unlink(temp_emo.name)
                    except OSError:
                        pass

        @api.post("/synthesize_with_url")
        async def synthesize_with_url(request: Request):
            """One-shot cloning: synthesize using a reference audio URL (no pre-registration)."""
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            # Rewrite to use /synthesize with reference_audio_url
            body.setdefault("reference_audio_url", body.pop("speaker_wav_url", None))
            from starlette.requests import Request as StarletteRequest
            from starlette.datastructures import State

            class FakeRequest:
                async def json(self_):
                    return body

            return await synthesize(FakeRequest())

        return api


@app.local_entrypoint()
def main():
    print("IndexTTS-2 service ready")
    print("Deploy with: modal deploy modal/indextts_service.py")
