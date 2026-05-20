"""
ModalModelService -- Base classes for Modal inference voice model deployments.

Eliminates boilerplate across Modal TTS/STT services by providing:
- Standard /synthesize, /synthesize_stream, /synthesize_with_url endpoints
- /clone_voice and /register_voice with audio_file or audio_url
- /voices listing and DELETE /voices/{voice_id}
- /health endpoint for gateway health checks
- Shared volume management for voice references
- Consistent error handling, timing headers, and logging
- WAV buffer serialization helpers

Subclass ModalTTSService for voice-cloning TTS models. Override:
    MODEL_NAME, GPU_TYPE, IMAGE, VOICE_VOLUME_NAME, SAMPLE_RATE,
    setup_model(), synthesize_impl(), and optionally synthesize_stream_impl().

Subclass ModalModelService for non-TTS models (STT, S2S, etc.) when you
only need the health/volume/error-handling scaffolding.

Example:
    class MyTTS(ModalTTSService):
        MODEL_NAME = "my-model"
        GPU_TYPE = "A10G"
        FEATURES = ["voice_cloning", "streaming"]

        def setup_model(self):
            from my_model import Model
            self.model = Model.load()

        def synthesize_impl(self, text, voice_id, **kwargs):
            ref_path = self.get_voice_ref_path(voice_id)
            audio = self.model.generate(text, ref=ref_path)
            return audio.numpy(), self.SAMPLE_RATE
"""

from __future__ import annotations

import io
import os
import time
import tempfile
import subprocess
import traceback
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers (importable without Modal -- useful in tests)
# ---------------------------------------------------------------------------

def wav_bytes_from_numpy(audio_np, sample_rate: int) -> bytes:
    """Serialize a numpy array to WAV bytes via soundfile."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def download_audio_to_tempfile(
    *,
    audio_url: str | None = None,
    audio_file=None,
    target_sr: int | None = None,
    mono: bool = True,
    max_duration: float | None = None,
    suffix: str = ".wav",
) -> str:
    """Download/convert audio to a temp WAV file. Returns the temp path.

    Caller is responsible for unlinking the file when done.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        raw_path = f.name

    if audio_url:
        cmd = [
            "ffmpeg", "-y",
            "-headers", "User-Agent: Mozilla/5.0",
            "-i", audio_url,
        ]
    elif audio_file is not None:
        # audio_file is bytes or has .read()
        data = audio_file if isinstance(audio_file, bytes) else audio_file
        with open(raw_path, "wb") as f:
            f.write(data)
        cmd = ["ffmpeg", "-y", "-i", raw_path]
    else:
        raise ValueError("Provide audio_url or audio_file")

    if target_sr:
        cmd += ["-ar", str(target_sr)]
    if mono:
        cmd += ["-ac", "1"]
    if max_duration:
        cmd += ["-t", str(max_duration)]
    cmd += ["-f", "wav", raw_path]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        if os.path.exists(raw_path):
            os.unlink(raw_path)
        raise RuntimeError(
            f"ffmpeg failed: {result.stderr.decode()[:500]}"
        )

    return raw_path


# ---------------------------------------------------------------------------
# ModalTTSService -- base for all voice-cloning TTS deployments
# ---------------------------------------------------------------------------

class ModalTTSService:
    """Base class for Modal inference TTS services on Modal.

    Class-level constants (override in subclasses):
        MODEL_NAME:          Short identifier, used in app name and logs.
        GPU_TYPE:            Modal GPU type ("A10G", "T4", None for CPU).
        SAMPLE_RATE:         Output audio sample rate (default 24000).
        VOICE_REF_SR:        Sample rate for stored voice references (default 16000).
        VOICE_REF_MAX_DUR:   Max duration in seconds for stored references (default 30).
        FEATURES:            List of feature strings for /health.
        CONTAINER_IDLE:      Container idle timeout seconds (default 180).
        CONCURRENT_INPUTS:   Max concurrent requests (default 5).

    Required overrides:
        setup_model()            -- load model weights, called once per container.
        synthesize_impl(text, voice_id, **kwargs)
                                 -- return (numpy_array, sample_rate).

    Optional overrides:
        synthesize_stream_impl(text, voice_id, **kwargs)
                                 -- yield numpy chunks; falls back to sentence
                                    splitting + synthesize_impl if not overridden.
        extra_routes(api)        -- add model-specific routes to the FastAPI app.
        health_extra()           -- return dict merged into /health response.
    """

    # -- Override these in subclasses ------------------------------------------
    MODEL_NAME: str = "base"
    GPU_TYPE: str | None = "A10G"
    SAMPLE_RATE: int = 24000
    VOICE_REF_SR: int = 16000
    VOICE_REF_MAX_DUR: float = 30
    FEATURES: list[str] = []
    CONTAINER_IDLE: int = 180
    CONCURRENT_INPUTS: int = 5

    # -- Internal state (set during lifecycle) ---------------------------------
    _voice_cache_volume = None  # Set by subclass module-level code

    # -- Lifecycle -------------------------------------------------------------

    def load_model(self):
        """Called by @modal.enter(). Delegates to setup_model() after creating dirs."""
        os.makedirs("/voice-cache/references", exist_ok=True)
        self.setup_model()

    def setup_model(self):
        """Override: load model weights here."""
        raise NotImplementedError

    # -- Voice reference helpers -----------------------------------------------

    def get_voice_ref_path(self, voice_id: str) -> str:
        """Return the on-disk path for a voice reference WAV."""
        return f"/voice-cache/references/{voice_id}.wav"

    def voice_exists(self, voice_id: str) -> bool:
        return os.path.exists(self.get_voice_ref_path(voice_id))

    def list_voice_ids(self) -> list[str]:
        ref_dir = "/voice-cache/references"
        if not os.path.exists(ref_dir):
            return []
        return [
            f[:-4] for f in os.listdir(ref_dir) if f.endswith(".wav")
        ]

    def commit_volume(self):
        """Commit voice cache volume changes. No-op if volume not set."""
        if self._voice_cache_volume is not None:
            self._voice_cache_volume.commit()

    # -- Synthesis interface ---------------------------------------------------

    def synthesize_impl(self, text: str, voice_id: str, **kwargs) -> tuple:
        """Override: synthesize text with the given voice.

        Args:
            text: Text to synthesize.
            voice_id: Voice reference identifier.
            **kwargs: Model-specific params (speed, exaggeration, etc.).

        Returns:
            Tuple of (audio_numpy_array, sample_rate).
        """
        raise NotImplementedError

    def synthesize_stream_impl(self, text: str, voice_id: str, **kwargs):
        """Override: yield (audio_numpy_array, sample_rate) chunks.

        Default implementation splits text into sentences and calls
        synthesize_impl per sentence.
        """
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [text]
        for sentence in sentences:
            yield self.synthesize_impl(sentence, voice_id, **kwargs)

    # -- FastAPI app builder ---------------------------------------------------

    def build_app(self):
        """Build and return a FastAPI app with standard TTS endpoints.

        Called from web_app() in the concrete Modal class.
        """
        from fastapi import FastAPI, Request, UploadFile, File, Form
        from fastapi.responses import Response, JSONResponse, StreamingResponse

        api = FastAPI()
        svc = self

        # -- /health -----------------------------------------------------------

        @api.get("/health")
        def health():
            task_type = "tts"
            if "stt" in (svc.FEATURES or []):
                task_type = "stt"
            elif "s2s" in (svc.FEATURES or []):
                task_type = "s2s"
            resp = {
                "status": "ok",
                "model": svc.MODEL_NAME,
                "gpu": svc.GPU_TYPE or "cpu",
                "sample_rate": svc.SAMPLE_RATE,
                "features": svc.FEATURES,
                # Standard models array for gateway health probes
                "models": [{
                    "name": svc.MODEL_NAME,
                    "slug": svc.MODEL_NAME,
                    "taskType": task_type,
                    "status": "ready",
                }],
            }
            extra = svc.health_extra()
            if extra:
                resp.update(extra)
            return resp

        # -- /voices -----------------------------------------------------------

        @api.get("/voices")
        def list_voices():
            voices = [
                {"id": vid, "type": "cloned"}
                for vid in svc.list_voice_ids()
            ]
            return {"voices": voices}

        # -- /clone_voice ------------------------------------------------------

        @api.post("/clone_voice")
        async def clone_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            if not audio_file and not audio_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Either audio_file or audio_url required"},
                )
            try:
                if audio_url:
                    audio_data = None
                    url = audio_url
                else:
                    audio_data = await audio_file.read()
                    url = None

                ref_path = svc.get_voice_ref_path(voice_id)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    if audio_data:
                        f.write(audio_data)

                cmd = ["ffmpeg", "-y"]
                if url:
                    cmd += ["-headers", "User-Agent: Mozilla/5.0", "-i", url]
                else:
                    cmd += ["-i", temp_path]
                cmd += [
                    "-ar", str(svc.VOICE_REF_SR),
                    "-ac", "1",
                    "-t", str(svc.VOICE_REF_MAX_DUR),
                    "-f", "wav", ref_path,
                ]

                result = subprocess.run(cmd, capture_output=True)

                # Clean up temp file (only if we wrote audio_data to it)
                if audio_data and os.path.exists(temp_path):
                    os.unlink(temp_path)

                if result.returncode != 0:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Failed to process audio"},
                    )

                svc.commit_volume()

                return {
                    "status": "success",
                    "voice_id": voice_id,
                    "message": f"Voice '{voice_id}' registered for cloning",
                }

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/register_voice")
        async def register_voice(
            voice_id: str = Form(...),
            audio_file: UploadFile = File(None),
            audio_url: str = Form(None),
        ):
            return await clone_voice(
                voice_id=voice_id, audio_file=audio_file, audio_url=audio_url,
            )

        # -- DELETE /voices/{voice_id} -----------------------------------------

        @api.delete("/voices/{voice_id}")
        def delete_voice(voice_id: str):
            ref_path = svc.get_voice_ref_path(voice_id)
            if os.path.exists(ref_path):
                os.unlink(ref_path)
                svc.commit_volume()
                return {"status": "deleted", "voice_id": voice_id}
            return JSONResponse(status_code=404, content={"error": "Voice not found"})

        # -- /synthesize -------------------------------------------------------

        @api.post("/synthesize")
        async def synthesize(request: Request):
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"},
                )

            # Accept both native (text/voice_id) and OpenAI-compat (input/voice)
            text = body.get("text") or body.get("input", "")
            voice_id = body.get("voice_id") or body.get("voice")

            if not text:
                return JSONResponse(
                    status_code=400, content={"error": "text is required"},
                )
            if not voice_id:
                return JSONResponse(
                    status_code=400, content={"error": "voice_id is required"},
                )

            if not svc.voice_exists(voice_id):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found"},
                )

            # Pass all extra body params to synthesize_impl
            extra = {
                k: v for k, v in body.items()
                if k not in ("text", "voice_id")
            }

            try:
                audio_np, sr = svc.synthesize_impl(text, voice_id, **extra)
                wav_data = wav_bytes_from_numpy(audio_np, sr)
                elapsed = time.time() - start_time

                print(
                    f"{svc.MODEL_NAME}: synthesized {len(text)} chars "
                    f"with '{voice_id}' in {elapsed:.3f}s"
                )

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Voice-Id": voice_id,
                        "X-Sample-Rate": str(sr),
                    },
                )

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        # -- /synthesize_stream ------------------------------------------------

        @api.post("/synthesize_stream")
        async def synthesize_stream(request: Request):
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"},
                )

            text = body.get("text", "")
            voice_id = body.get("voice_id")

            if not text:
                return JSONResponse(
                    status_code=400, content={"error": "text is required"},
                )
            if not voice_id:
                return JSONResponse(
                    status_code=400, content={"error": "voice_id is required"},
                )

            if not svc.voice_exists(voice_id):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found"},
                )

            extra = {
                k: v for k, v in body.items()
                if k not in ("text", "voice_id")
            }

            def generate():
                first = True
                for audio_np, sr in svc.synthesize_stream_impl(
                    text, voice_id, **extra
                ):
                    data = wav_bytes_from_numpy(audio_np, sr)
                    if first:
                        yield data
                        first = False
                    else:
                        # Skip 44-byte WAV header for subsequent chunks
                        yield data[44:]

            return StreamingResponse(
                generate(),
                media_type="audio/wav",
                headers={"X-Voice-Id": voice_id},
            )

        # -- /synthesize_with_url (one-shot cloning) ---------------------------

        @api.post("/synthesize_with_url")
        async def synthesize_with_url(request: Request):
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"},
                )

            text = body.get("text", "")
            speaker_wav_url = body.get("speaker_wav_url")

            if not text:
                return JSONResponse(
                    status_code=400, content={"error": "text is required"},
                )
            if not speaker_wav_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "speaker_wav_url is required"},
                )

            extra = {
                k: v for k, v in body.items()
                if k not in ("text", "speaker_wav_url")
            }

            temp_path = None
            try:
                temp_path = download_audio_to_tempfile(
                    audio_url=speaker_wav_url,
                    target_sr=svc.VOICE_REF_SR,
                    max_duration=svc.VOICE_REF_MAX_DUR,
                )

                audio_np, sr = svc.synthesize_with_url_impl(
                    text, temp_path, **extra
                )
                wav_data = wav_bytes_from_numpy(audio_np, sr)
                elapsed = time.time() - start_time

                print(
                    f"{svc.MODEL_NAME}: one-shot synthesis in {elapsed:.3f}s"
                )

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={"X-Generation-Time": f"{elapsed:.3f}"},
                )

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        # -- Model-specific routes ---------------------------------------------

        svc.extra_routes(api)

        return api

    # -- Extension points ------------------------------------------------------

    def health_extra(self) -> dict | None:
        """Override to add extra fields to /health response."""
        return None

    def extra_routes(self, api):
        """Override to add model-specific routes to the FastAPI app."""
        pass

    def synthesize_with_url_impl(
        self, text: str, ref_audio_path: str, **kwargs
    ) -> tuple:
        """One-shot synthesis from a temporary reference audio path.

        Default implementation copies the ref to a temp voice_id, synthesizes,
        then cleans up. Override if your model has a direct one-shot API.

        Returns:
            Tuple of (audio_numpy_array, sample_rate).
        """
        import uuid

        temp_voice_id = f"_oneshot_{uuid.uuid4().hex[:8]}"
        ref_path = self.get_voice_ref_path(temp_voice_id)

        try:
            # Copy the downloaded file to the voice ref location
            subprocess.run(
                ["cp", ref_audio_path, ref_path],
                check=True,
            )
            return self.synthesize_impl(text, temp_voice_id, **kwargs)
        finally:
            if os.path.exists(ref_path):
                os.unlink(ref_path)


# ---------------------------------------------------------------------------
# Modal decorator helpers
# ---------------------------------------------------------------------------

def create_modal_app(model_name: str):
    """Create a Modal App with the standard Modal inference naming convention."""
    import modal
    return modal.App(f"modal-inference-{model_name}")


def base_service_layer(image):
    """Add base_service.py to a Modal Image so it's available at build time.

    Must wrap the base image BEFORE any .run_function() or .run_commands()
    that import base_service.

    Usage:
        image = (
            base_service_layer(modal.Image.debian_slim(python_version="3.11"))
            .apt_install("ffmpeg")
            .pip_install(...)
            .run_function(download_model)  # can now import base_service
        )
    """
    import pathlib
    base_path = str(pathlib.Path(__file__).resolve())
    return image.add_local_file(base_path, remote_path="/root/base_service.py", copy=True)


def create_voice_volume(model_name: str):
    """Create or get a Modal Volume for voice references."""
    import modal
    return modal.Volume.from_name(
        f"phony-{model_name}-voices", create_if_missing=True,
    )


def create_model_volume():
    """Create or get the shared model weight cache volume."""
    import modal
    return modal.Volume.from_name("phony-model-cache", create_if_missing=True)
