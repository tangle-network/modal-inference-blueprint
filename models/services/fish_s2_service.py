"""
Fish Audio S2 Pro TTS Service - SOTA open-source TTS with emotion control.

Fish Audio S2 Pro is a 4B-parameter TTS model with the lowest WER on
Seed-TTS Eval. Supports inline emotion tags, 80+ languages, and zero-shot
voice cloning from 10-30s reference audio.

The model uses a three-stage pipeline:
1. Encode reference audio -> VQ tokens (via DAC codec)
2. Generate semantic tokens from text + ref tokens (via LLaMA-based model)
3. Decode semantic tokens -> audio waveform (via DAC codec)

We wrap this as a subprocess pipeline since fish-speech's internal Python
API is CLI-oriented. The subprocess calls are colocated on the same GPU
container so latency is minimal.

Capabilities:
- Zero-shot voice cloning from 10-30s reference audio
- Inline emotion tags: [whisper], [laugh], [excited], [professional], [pause]
- 80+ languages
- 4B parameters, Dual-Autoregressive architecture
- Apache 2.0 license (code), CC-BY-NC-SA-4.0 (model weights)

Deploy: modal deploy modal/fish_s2_service.py
"""

import modal
import os

from base_service import (
    base_service_layer,
    PhonyTTSService,
    create_modal_app,
    create_voice_volume,
)

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("fish-s2")
voice_cache = create_voice_volume("fish-s2")
model_cache = modal.Volume.from_name("phony-model-cache", create_if_missing=True)

# --- Container image ---------------------------------------------------------
# Fish Speech S2 Pro requires the fish-speech repo + its dependencies.
# The model is ~9GB; we bake it into the image for fast cold starts.

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.12"))
    .apt_install("ffmpeg", "git", "git-lfs", "libsox-dev", "portaudio19-dev")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "transformers>=4.40",
        "huggingface_hub[cli]",
        "requests",
    )
    .run_commands(
        # Clone fish-speech repo and install with CUDA support
        "git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech",
        "cd /opt/fish-speech && pip install -e '.[cu126]'",
    )
    .run_commands(
        # Download S2 Pro model weights (~9GB)
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('fishaudio/s2-pro', "
        "local_dir='/opt/fish-speech/checkpoints/s2-pro')"
        "\"",
    )
)

CHECKPOINT_DIR = "/opt/fish-speech/checkpoints/s2-pro"
CODEC_PATH = f"{CHECKPOINT_DIR}/codec.pth"
REF_TOKENS_DIR = "/voice-cache/ref-tokens"


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=180,
    volumes={
        "/voice-cache": voice_cache,
        "/model-cache": model_cache,
    },
    allow_concurrent_inputs=3,  # 4B model uses ~17GB VRAM per request
)
class FishS2Service(PhonyTTSService):
    """Fish Audio S2 Pro TTS with zero-shot cloning and emotion tags."""

    MODEL_NAME = "fish-s2-pro"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 44100  # Fish Speech DAC codec outputs 44.1kHz
    VOICE_REF_SR = 44100  # Keep ref audio at native rate for best quality
    VOICE_REF_MAX_DUR = 30  # 10-30s recommended
    FEATURES = [
        "voice_cloning",
        "zero_shot",
        "emotion_tags",
        "multilingual",
        "80_plus_languages",
    ]
    CONTAINER_IDLE = 180
    CONCURRENT_INPUTS = 3

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        """Load Fish Speech S2 Pro model components."""
        import sys
        sys.path.insert(0, "/opt/fish-speech")

        os.makedirs(REF_TOKENS_DIR, exist_ok=True)

        # Try the high-level TTS API first (available in newer fish-speech)
        self._tts_api = None
        self._use_subprocess = False

        try:
            import torch
            from fish_speech.tts.api import TTS

            self._tts_api = TTS(
                llama_path=CHECKPOINT_DIR,
                decoder_path=CODEC_PATH,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"Fish S2 Pro loaded via TTS API from {CHECKPOINT_DIR}")
        except (ImportError, AttributeError, TypeError) as e:
            # Fall back to subprocess pipeline if TTS API not available
            print(f"TTS API not available ({e}), using subprocess pipeline")
            self._use_subprocess = True
            # Verify the checkpoint files exist
            if not os.path.exists(CODEC_PATH):
                raise RuntimeError(
                    f"Codec checkpoint not found at {CODEC_PATH}. "
                    "Model download may have failed during image build."
                )
            print(f"Fish S2 Pro ready (subprocess mode) from {CHECKPOINT_DIR}")

    def _get_ref_tokens_path(self, voice_id: str) -> str:
        """Path to cached reference tokens for a voice."""
        return os.path.join(REF_TOKENS_DIR, f"{voice_id}.npy")

    def _encode_reference_subprocess(self, audio_path: str, tokens_path: str):
        """Encode reference audio to VQ tokens via the DAC codec CLI."""
        import subprocess

        cmd = [
            "python", "/opt/fish-speech/fish_speech/models/dac/inference.py",
            "-i", audio_path,
            "--checkpoint-path", CODEC_PATH,
            "-o", tokens_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, cwd="/opt/fish-speech"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Reference encoding failed: {result.stderr.decode()[:500]}"
            )

    def _ensure_ref_tokens(self, voice_id: str) -> str:
        """Ensure reference tokens exist for a voice, encoding if needed."""
        tokens_path = self._get_ref_tokens_path(voice_id)
        if os.path.exists(tokens_path):
            return tokens_path

        ref_path = self.get_voice_ref_path(voice_id)
        self._encode_reference_subprocess(ref_path, tokens_path)
        self.commit_volume()
        return tokens_path

    def health_extra(self):
        return {
            "model_id": "fishaudio/s2-pro",
            "params": "4B",
            "architecture": "Dual-Autoregressive",
            "inference_mode": "api" if not self._use_subprocess else "subprocess",
            "emotion_tags": [
                "[whisper]", "[laugh]", "[excited]",
                "[professional]", "[pause]", "[sad]",
            ],
        }

    def synthesize_impl(self, text: str, voice_id: str, **kwargs) -> tuple:
        """Synthesize speech with Fish S2 Pro.

        Inline emotion tags can be embedded directly in text:
            "Hello [whisper] this is a secret [laugh]"

        Extra kwargs:
            prompt_text (str): Transcript of the reference audio for better
                cloning accuracy. Optional but recommended.
        """
        if not self._use_subprocess:
            return self._synthesize_api(text, voice_id, **kwargs)
        return self._synthesize_subprocess(text, voice_id, **kwargs)

    def _synthesize_api(self, text: str, voice_id: str, **kwargs) -> tuple:
        """Synthesis via the high-level fish_speech.tts.api.TTS class."""
        ref_path = self.get_voice_ref_path(voice_id)
        audio_data = self._tts_api.text_to_speech(
            text=text,
            reference_audio_path=ref_path,
        )
        return audio_data, self.SAMPLE_RATE

    def _synthesize_subprocess(self, text: str, voice_id: str, **kwargs) -> tuple:
        """Three-stage subprocess synthesis pipeline."""
        import subprocess
        import tempfile
        import numpy as np
        import soundfile as sf

        prompt_text = kwargs.get("prompt_text", "")
        tokens_path = self._ensure_ref_tokens(voice_id)

        # Stage 1: Generate semantic tokens from text + ref tokens
        with tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, dir="/tmp"
        ) as f:
            codes_path = f.name

        try:
            cmd = [
                "python",
                "/opt/fish-speech/fish_speech/models/text2semantic/inference.py",
                "--text", text,
                "--prompt-tokens", tokens_path,
                "--checkpoint-path", CHECKPOINT_DIR,
                "-o", codes_path,
            ]
            if prompt_text:
                cmd += ["--prompt-text", prompt_text]

            result = subprocess.run(
                cmd, capture_output=True, cwd="/opt/fish-speech"
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Semantic generation failed: {result.stderr.decode()[:500]}"
                )

            # Stage 2: Decode semantic tokens to audio
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir="/tmp"
            ) as f:
                output_wav = f.name

            cmd = [
                "python",
                "/opt/fish-speech/fish_speech/models/dac/inference.py",
                "-i", codes_path,
                "--checkpoint-path", CODEC_PATH,
                "-o", output_wav,
            ]
            result = subprocess.run(
                cmd, capture_output=True, cwd="/opt/fish-speech"
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Audio decoding failed: {result.stderr.decode()[:500]}"
                )

            # Read the generated audio
            audio_np, sr = sf.read(output_wav)
            return audio_np, sr

        finally:
            for p in [codes_path, output_wav if "output_wav" in dir() else None]:
                if p and os.path.exists(p):
                    os.unlink(p)

    def synthesize_with_url_impl(self, text: str, ref_audio_path: str, **kwargs) -> tuple:
        """One-shot synthesis directly from a temp reference audio path."""
        if not self._use_subprocess:
            audio_data = self._tts_api.text_to_speech(
                text=text,
                reference_audio_path=ref_audio_path,
            )
            return audio_data, self.SAMPLE_RATE

        # Subprocess: encode the temp ref, then synthesize
        import subprocess
        import tempfile
        import soundfile as sf

        prompt_text = kwargs.get("prompt_text", "")

        with tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, dir="/tmp"
        ) as f:
            ref_tokens_path = f.name

        try:
            self._encode_reference_subprocess(ref_audio_path, ref_tokens_path)

            with tempfile.NamedTemporaryFile(
                suffix=".npy", delete=False, dir="/tmp"
            ) as f:
                codes_path = f.name

            cmd = [
                "python",
                "/opt/fish-speech/fish_speech/models/text2semantic/inference.py",
                "--text", text,
                "--prompt-tokens", ref_tokens_path,
                "--checkpoint-path", CHECKPOINT_DIR,
                "-o", codes_path,
            ]
            if prompt_text:
                cmd += ["--prompt-text", prompt_text]

            result = subprocess.run(
                cmd, capture_output=True, cwd="/opt/fish-speech"
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Semantic generation failed: {result.stderr.decode()[:500]}"
                )

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir="/tmp"
            ) as f:
                output_wav = f.name

            cmd = [
                "python",
                "/opt/fish-speech/fish_speech/models/dac/inference.py",
                "-i", codes_path,
                "--checkpoint-path", CODEC_PATH,
                "-o", output_wav,
            ]
            result = subprocess.run(
                cmd, capture_output=True, cwd="/opt/fish-speech"
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Audio decoding failed: {result.stderr.decode()[:500]}"
                )

            audio_np, sr = sf.read(output_wav)
            return audio_np, sr

        finally:
            for p in [ref_tokens_path, codes_path, output_wav]:
                if p and os.path.exists(p):
                    os.unlink(p)

    def extra_routes(self, api):
        """Add Fish S2-specific routes for emotion tags and ref token cache."""
        from fastapi import Request
        from fastapi.responses import JSONResponse
        import time

        svc = self

        @api.get("/emotion_tags")
        def emotion_tags():
            """List supported inline emotion tags."""
            return {
                "tags": [
                    {"tag": "[whisper]", "description": "Whispered speech"},
                    {"tag": "[laugh]", "description": "Laughter"},
                    {"tag": "[excited]", "description": "Excited/energetic tone"},
                    {"tag": "[professional]", "description": "Professional/formal tone"},
                    {"tag": "[pause]", "description": "Insert a pause"},
                    {"tag": "[sad]", "description": "Sad/melancholic tone"},
                    {"tag": "[angry]", "description": "Angry tone"},
                    {"tag": "[happy]", "description": "Happy/cheerful tone"},
                ],
                "usage": "Embed tags inline in text: 'Hello [whisper] this is a secret [laugh]'",
            }

        @api.post("/warmup")
        async def warmup(request: Request):
            """Pre-encode a voice reference into tokens (subprocess mode)."""
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    status_code=400, content={"error": "Invalid JSON"}
                )

            voice_id = body.get("voice_id")
            if not voice_id:
                return JSONResponse(
                    status_code=400, content={"error": "voice_id is required"}
                )

            if not svc.voice_exists(voice_id):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found"},
                )

            start = time.time()
            try:
                # Force re-encode
                tokens_path = svc._get_ref_tokens_path(voice_id)
                if os.path.exists(tokens_path):
                    os.unlink(tokens_path)
                svc._ensure_ref_tokens(voice_id)
                elapsed = time.time() - start
                print(f"Fish S2: warmed up voice '{voice_id}' in {elapsed:.2f}s")
                return {
                    "status": "warmed",
                    "voice_id": voice_id,
                    "warmup_time": elapsed,
                }
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/cache/clear")
        async def clear_cache():
            """Clear all cached reference tokens."""
            count = 0
            if os.path.exists(REF_TOKENS_DIR):
                for f in os.listdir(REF_TOKENS_DIR):
                    if f.endswith(".npy"):
                        os.unlink(os.path.join(REF_TOKENS_DIR, f))
                        count += 1
            svc.commit_volume()
            return {"status": "cleared", "evicted": count}

    @modal.asgi_app()
    def web_app(self):
        return self.build_app()


@app.local_entrypoint()
def main():
    print("Fish Audio S2 Pro TTS service ready")
    print("Deploy with: modal deploy modal/fish_s2_service.py")
