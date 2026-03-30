"""
CosyVoice3 TTS Service - Alibaba's next-gen zero-shot voice cloning.

CosyVoice3 (Fun-CosyVoice3-0.5B) is a 1.5B-parameter multilingual TTS model
with zero-shot voice cloning, 9 languages, 18+ Chinese dialects, and streaming.
Major upgrade over CosyVoice2 with better stability, accuracy, and speed.

Capabilities:
- Zero-shot voice cloning from 3-10s reference audio
- Streaming output (native)
- 9 languages + 18 Chinese dialects
- Instruction-controlled speech style (speed, emotion)
- ~150-300ms TTFB on A10G
- Apache 2.0 licensed (code), model weights may have separate license

Deploy: modal deploy modal/cosyvoice3_service.py
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

app = create_modal_app("cosyvoice3")
voice_cache = create_voice_volume("cosyvoice3")
model_cache = modal.Volume.from_name("phony-model-cache", create_if_missing=True)

# --- Container image ---------------------------------------------------------
# CosyVoice3 requires the full repo install (cosyvoice pip package + matcha-tts).
# We pin Python 3.10 because ttsfrd wheels target cp310.

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.10"))
    .apt_install("ffmpeg", "git", "sox", "libsox-dev")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "transformers>=4.40",
        "onnxruntime",
        "conformer",
        "lightning",
        "inflect",
        "pyyaml",
        "hyperpyyaml",
        "openai-whisper",
        "wget",
        "huggingface_hub",
    )
    .run_commands(
        # Install CosyVoice from source (includes v3 AutoModel)
        "pip install git+https://github.com/FunAudioLLM/CosyVoice.git",
        "pip install matcha-tts",
    )
    .run_commands(
        # Pre-download model weights into the image so cold starts are fast.
        # Fun-CosyVoice3-0.5B-2512 is the latest v3 checkpoint.
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', "
        "local_dir='/root/pretrained_models/Fun-CosyVoice3-0.5B')"
        "\"",
    )
)


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
    allow_concurrent_inputs=5,
)
class CosyVoice3Service(PhonyTTSService):
    """CosyVoice3 TTS with zero-shot voice cloning (1.5B params, 9 languages)."""

    MODEL_NAME = "cosyvoice3"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 24000
    VOICE_REF_SR = 16000
    VOICE_REF_MAX_DUR = 15  # CosyVoice works best with 3-10s refs
    FEATURES = [
        "voice_cloning",
        "zero_shot",
        "streaming",
        "multilingual",
        "instruction_control",
    ]

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        """Load CosyVoice3 model weights."""
        import sys
        sys.path.append("third_party/Matcha-TTS")

        from cosyvoice.cli.cosyvoice import AutoModel

        model_dir = "/root/pretrained_models/Fun-CosyVoice3-0.5B"
        self.model = AutoModel(model_dir=model_dir)
        self._sample_rate = self.model.sample_rate
        print(f"CosyVoice3 loaded from {model_dir} (sr={self._sample_rate})")

    def health_extra(self):
        return {
            "model_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            "params": "1.5B",
            "languages": 9,
            "chinese_dialects": 18,
        }

    def _run_zero_shot(self, text: str, prompt_text: str, ref_path: str, stream: bool = False):
        """Call inference_zero_shot with the v3 AutoModel positional API.

        CosyVoice3 AutoModel.inference_zero_shot signature:
            inference_zero_shot(tts_text, prompt_text, prompt_wav_path, stream=False)
        Returns an iterator of dicts with key 'tts_speech' (torch tensors).
        """
        return self.model.inference_zero_shot(
            text,          # tts_text
            prompt_text,   # prompt_text (instruction + endofprompt token)
            ref_path,      # prompt_wav_path (file path, not tensor)
            stream=stream,
        )

    def synthesize_impl(self, text: str, voice_id: str, **kwargs) -> tuple:
        """Zero-shot synthesis using a registered voice reference.

        Extra kwargs:
            prompt_text (str): Instruction/prompt text for style control.
                Default includes endofprompt token for natural generation.
                Speed/emotion can be controlled via instruction text, e.g.
                "Speak slowly with a warm tone.<|endofprompt|>"
        """
        import torch

        prompt_text = kwargs.get(
            "prompt_text",
            "You are a helpful assistant.<|endofprompt|>",
        )
        ref_path = self.get_voice_ref_path(voice_id)

        audio_chunks = []
        for result in self._run_zero_shot(text, prompt_text, ref_path, stream=False):
            audio_chunks.append(result["tts_speech"])

        full_audio = torch.cat(audio_chunks, dim=-1).squeeze().cpu().numpy()
        return full_audio, self._sample_rate

    def synthesize_stream_impl(self, text: str, voice_id: str, **kwargs):
        """Native streaming: yield chunks as CosyVoice3 generates them."""
        prompt_text = kwargs.get(
            "prompt_text",
            "You are a helpful assistant.<|endofprompt|>",
        )
        ref_path = self.get_voice_ref_path(voice_id)

        for result in self._run_zero_shot(text, prompt_text, ref_path, stream=True):
            audio_np = result["tts_speech"].squeeze().cpu().numpy()
            yield audio_np, self._sample_rate

    def synthesize_with_url_impl(self, text: str, ref_audio_path: str, **kwargs) -> tuple:
        """One-shot synthesis: use the temp ref file path directly."""
        import torch

        prompt_text = kwargs.get(
            "prompt_text",
            "You are a helpful assistant.<|endofprompt|>",
        )

        audio_chunks = []
        for result in self._run_zero_shot(text, prompt_text, ref_audio_path, stream=False):
            audio_chunks.append(result["tts_speech"])

        full_audio = torch.cat(audio_chunks, dim=-1).squeeze().cpu().numpy()
        return full_audio, self._sample_rate

    def extra_routes(self, api):
        """Add CosyVoice3-specific instruction synthesis route."""
        from fastapi import Request
        from fastapi.responses import Response, JSONResponse
        import time

        from base_service import wav_bytes_from_numpy

        svc = self

        @api.post("/synthesize_instruct")
        async def synthesize_instruct(request: Request):
            """Instruction-controlled synthesis.

            Request body:
            {
                "text": "Hello world",
                "voice_id": "huberman",
                "instruct_text": "Speak slowly with a warm tone"
            }

            Speed, emotion, and style are controlled entirely via
            instruct_text which becomes the prompt instruction.
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"}
                )

            text = body.get("text", "")
            voice_id = body.get("voice_id")
            instruct_text = body.get("instruct_text", "")

            if not text:
                return JSONResponse(
                    status_code=400, content={"error": "text is required"}
                )
            if not voice_id:
                return JSONResponse(
                    status_code=400, content={"error": "voice_id is required"}
                )
            if not svc.voice_exists(voice_id):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found"},
                )

            try:
                import torch

                ref_path = svc.get_voice_ref_path(voice_id)

                # Build instruction prompt for style control
                prompt_text = (
                    f"You are a helpful assistant.<|endofprompt|>{instruct_text}"
                    if instruct_text
                    else "You are a helpful assistant.<|endofprompt|>"
                )

                audio_chunks = []
                for result in svc._run_zero_shot(
                    text, prompt_text, ref_path, stream=False
                ):
                    audio_chunks.append(result["tts_speech"])

                full_audio = (
                    torch.cat(audio_chunks, dim=-1).squeeze().cpu().numpy()
                )
                wav_data = wav_bytes_from_numpy(full_audio, svc._sample_rate)
                elapsed = time.time() - start_time

                print(
                    f"CosyVoice3: instruct synthesis {len(text)} chars "
                    f"with '{voice_id}' in {elapsed:.3f}s"
                )

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Voice-Id": voice_id,
                        "X-Sample-Rate": str(svc._sample_rate),
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
    print("CosyVoice3 TTS service ready")
    print("Deploy with: modal deploy modal/cosyvoice3_service.py")
