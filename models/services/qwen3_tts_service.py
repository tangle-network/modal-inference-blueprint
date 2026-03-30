"""
Qwen3-TTS Service - Alibaba's multilingual TTS with voice cloning and voice design.

Capabilities:
- 10+ languages, 49+ preset voices (CustomVoice model)
- 3-second voice cloning from reference audio (Base model)
- Natural language voice design instructions
- Dual-track streaming/non-streaming architecture
- 0.6B and 1.7B parameter variants
- Apache-2.0 license

Models loaded:
- Qwen3-TTS-12Hz-1.7B-Base (voice cloning)
- Qwen3-TTS-12Hz-1.7B-CustomVoice (preset voices + instruction control)

Deploy: modal deploy modal/qwen3_tts_service.py
"""

import modal
import os

from base_service import (
    base_service_layer,
    PhonyTTSService,
    create_modal_app,
    create_voice_volume,
    wav_bytes_from_numpy,
)

app = create_modal_app("qwen3-tts")
voice_cache = create_voice_volume("qwen3-tts")

MODEL_DIR = "/models/qwen3-tts"

PRESET_SPEAKERS = [
    "Chelsie", "Ethan", "Ryan", "Alya", "Layla",
    "Tyler", "Aiden", "Sophia", "Marcus", "Nova",
]

SUPPORTED_LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", "French",
    "German", "Spanish", "Italian", "Portuguese", "Russian",
]


def download_models():
    """Download Qwen3-TTS models at image build time."""
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    for repo in [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ]:
        print(f"Downloading {repo}...")
        snapshot_download(
            repo_id=repo,
            local_dir=f"{MODEL_DIR}/{repo.split('/')[-1]}",
        )
    print("Qwen3-TTS models downloaded")


image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "git", "sox", "libsndfile1")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "qwen-tts",
        "flash-attn",
        "accelerate",
        "transformers>=4.57.3",
        "huggingface_hub",
    )
    .run_function(download_models)
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=180,
    volumes={"/voice-cache": voice_cache},
    allow_concurrent_inputs=5,
)
class Qwen3TTSService(PhonyTTSService):
    """Qwen3-TTS with voice cloning (Base) and preset voices (CustomVoice)."""

    MODEL_NAME = "qwen3-tts"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 24000
    VOICE_REF_SR = 24000
    VOICE_REF_MAX_DUR = 15
    FEATURES = [
        "voice_cloning", "voice_design", "preset_voices",
        "multilingual", "streaming",
    ]

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        import torch

        self._models = {}
        self._clone_prompt_cache = {}
        self._max_prompt_cache = 20

        from qwen_tts import Qwen3TTSModel

        # Load Base model (voice cloning)
        self._models["base"] = Qwen3TTSModel.from_pretrained(
            f"{MODEL_DIR}/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        # Load CustomVoice model (preset speakers + instruction)
        self._models["custom"] = Qwen3TTSModel.from_pretrained(
            f"{MODEL_DIR}/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

        print("Qwen3-TTS Base + CustomVoice models loaded")

    def _get_or_create_clone_prompt(self, voice_id: str):
        """Build and cache a reusable voice clone prompt from reference audio."""
        if voice_id in self._clone_prompt_cache:
            return self._clone_prompt_cache[voice_id]

        ref_path = self.get_voice_ref_path(voice_id)
        model = self._models["base"]

        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_path,
            ref_text="",
            x_vector_only_mode=True,
        )

        # Evict oldest if cache full
        if len(self._clone_prompt_cache) >= self._max_prompt_cache:
            oldest = next(iter(self._clone_prompt_cache))
            del self._clone_prompt_cache[oldest]

        self._clone_prompt_cache[voice_id] = prompt
        return prompt

    def health_extra(self):
        return {
            "loaded_models": list(self._models.keys()),
            "preset_speakers": PRESET_SPEAKERS,
            "supported_languages": SUPPORTED_LANGUAGES,
            "version": "1.7B",
        }

    def synthesize_impl(self, text, voice_id, **kwargs):
        import numpy as np

        language = kwargs.get("language", "English")
        speaker = kwargs.get("speaker")
        instruct = kwargs.get("instruct")
        mode = kwargs.get("mode", "clone")

        # Preset voice via CustomVoice model
        if mode == "preset" and speaker:
            model = self._models["custom"]
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                **({"instruct": instruct} if instruct else {}),
            )
            return np.array(wavs[0]), sr

        # Voice cloning via Base model
        clone_prompt = self._get_or_create_clone_prompt(voice_id)
        model = self._models["base"]

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=clone_prompt,
        )
        return np.array(wavs[0]), sr

    def synthesize_stream_impl(self, text, voice_id, **kwargs):
        """Sentence-level streaming using clone prompt caching."""
        import re
        import numpy as np

        language = kwargs.get("language", "English")
        clone_prompt = self._get_or_create_clone_prompt(voice_id)
        model = self._models["base"]

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [text]

        for sentence in sentences:
            wavs, sr = model.generate_voice_clone(
                text=sentence,
                language=language,
                voice_clone_prompt=clone_prompt,
            )
            yield np.array(wavs[0]), sr

    def synthesize_with_url_impl(self, text, ref_audio_path, **kwargs):
        """One-shot voice cloning from a temporary reference audio file."""
        import numpy as np

        language = kwargs.get("language", "English")
        model = self._models["base"]

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text="",
            x_vector_only_mode=True,
        )
        return np.array(wavs[0]), sr

    def extra_routes(self, api):
        from fastapi import Request
        from fastapi.responses import JSONResponse, Response
        import time

        svc = self

        @api.get("/speakers")
        def list_speakers():
            return {
                "speakers": PRESET_SPEAKERS,
                "languages": SUPPORTED_LANGUAGES,
            }

        @api.post("/synthesize_preset")
        async def synthesize_preset(request: Request):
            """Synthesize with a preset speaker (no voice reference needed)."""
            start = time.time()
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            speaker = body.get("speaker", "Ryan")
            language = body.get("language", "English")
            instruct = body.get("instruct")

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            try:
                audio_np, sr = svc.synthesize_impl(
                    text, voice_id="",
                    speaker=speaker, language=language,
                    instruct=instruct, mode="preset",
                )
                wav_data = wav_bytes_from_numpy(audio_np, sr)
                elapsed = time.time() - start

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Speaker": speaker,
                        "X-Sample-Rate": str(sr),
                    },
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

    @modal.asgi_app()
    def web_app(self):
        return self.build_app()


@app.local_entrypoint()
def main():
    print("Qwen3-TTS service ready (Base 1.7B + CustomVoice 1.7B)")
    print("Deploy with: modal deploy modal/qwen3_tts_service.py")
