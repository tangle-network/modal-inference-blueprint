"""
Dia 1.6B TTS Service - Nari Labs ultra-realistic dialogue generation.

Capabilities:
- 1.6B params, single-pass dialogue generation
- Two-speaker dialogue with [S1]/[S2] tags
- Nonverbal sounds: (laughs), (coughs), (sighs), (gasps), etc.
- Voice conditioning from 5-10s audio prompt
- 44.1kHz output, SoundStorm-inspired architecture
- English only
- Apache 2.0 licensed

Deploy: modal deploy modal/dia_service.py
"""

import modal
import os

from base_service import (
    base_service_layer,
    ModalTTSService,
    create_modal_app,
    create_voice_volume,
    create_model_volume,
)

app = create_modal_app("dia-tts")
voice_cache = create_voice_volume("dia")
model_cache = create_model_volume()

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
    )
    .run_commands(
        "pip install git+https://github.com/nari-labs/dia.git"
    )
)


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
    secrets=[modal.Secret.from_name("huggingface")],
)
class DiaService(ModalTTSService):
    """Dia 1.6B — ultra-realistic dialogue TTS."""

    MODEL_NAME = "dia-tts"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 44100
    VOICE_REF_SR = 44100
    VOICE_REF_MAX_DUR = 15
    FEATURES = [
        "dialogue", "nonverbal", "voice_cloning", "two_speaker",
    ]

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        from dia.model import Dia

        cache_dir = "/model-cache/dia"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir

        self.model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626",
            compute_dtype="float16",
        )
        print("Dia 1.6B loaded")

    def synthesize_impl(self, text: str, voice_id: str, **kwargs):
        """Synthesize dialogue text with optional voice conditioning.

        Text should use [S1] and [S2] tags for speaker turns.
        If no tags present, auto-wraps with [S1].

        Extra kwargs:
            cfg_scale (float): Classifier-free guidance scale (default 3.0)
            temperature (float): Sampling temperature (default 1.2)
            top_p (float): Nucleus sampling threshold (default 0.95)
            max_tokens (int): Max generation tokens (default 3072)
        """
        cfg_scale = kwargs.get("cfg_scale", 3.0)
        temperature = kwargs.get("temperature", 1.2)
        top_p = kwargs.get("top_p", 0.95)
        max_tokens = kwargs.get("max_tokens", 3072)

        # Auto-wrap text with [S1] tag if no speaker tags present
        if "[S1]" not in text and "[S2]" not in text:
            text = f"[S1] {text}"

        # Voice conditioning: use registered voice ref as audio prompt
        ref_path = self.get_voice_ref_path(voice_id)
        audio_prompt_path = ref_path if os.path.exists(ref_path) else None

        output = self.model.generate(
            text=text,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=45,
            audio_prompt_path=audio_prompt_path,
            verbose=False,
        )

        # output is np.ndarray
        import numpy as np
        if isinstance(output, list):
            output = output[0]
        audio_np = np.asarray(output, dtype=np.float32)

        # Normalize to [-1, 1] if needed
        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            peak = max(abs(audio_np.max()), abs(audio_np.min()))
            if peak > 0:
                audio_np = audio_np / peak

        return audio_np, self.SAMPLE_RATE

    def synthesize_with_url_impl(self, text: str, ref_audio_path: str, **kwargs):
        """One-shot synthesis with voice conditioning from URL audio."""
        cfg_scale = kwargs.get("cfg_scale", 3.0)
        temperature = kwargs.get("temperature", 1.2)
        top_p = kwargs.get("top_p", 0.95)
        max_tokens = kwargs.get("max_tokens", 3072)

        if "[S1]" not in text and "[S2]" not in text:
            text = f"[S1] {text}"

        import numpy as np

        output = self.model.generate(
            text=text,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=45,
            audio_prompt_path=ref_audio_path,
            verbose=False,
        )

        if isinstance(output, list):
            output = output[0]
        audio_np = np.asarray(output, dtype=np.float32)

        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            peak = max(abs(audio_np.max()), abs(audio_np.min()))
            if peak > 0:
                audio_np = audio_np / peak

        return audio_np, self.SAMPLE_RATE

    def health_extra(self):
        return {
            "version": "1.6B-0626",
            "speaker_tags": ["[S1]", "[S2]"],
            "nonverbal_tags": [
                "(laughs)", "(coughs)", "(sighs)", "(gasps)",
                "(clears throat)", "(singing)",
            ],
        }

    def extra_routes(self, api):
        from fastapi import Request
        from fastapi.responses import Response, JSONResponse
        import time

        svc = self

        @api.post("/synthesize_dialogue")
        async def synthesize_dialogue(request: Request):
            """Synthesize multi-speaker dialogue.

            Body:
                text: Dialogue with [S1]/[S2] tags
                voice_id: Optional voice conditioning reference
                cfg_scale, temperature, top_p, max_tokens: Generation params
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

            if "[S1]" not in text:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Dialogue text must contain [S1] speaker tags"},
                )

            extra = {k: v for k, v in body.items() if k not in ("text", "voice_id")}

            try:
                if voice_id and svc.voice_exists(voice_id):
                    audio_np, sr = svc.synthesize_impl(text, voice_id, **extra)
                else:
                    # No voice conditioning — synthesize without ref
                    import numpy as np

                    if "[S1]" not in text and "[S2]" not in text:
                        text = f"[S1] {text}"

                    output = svc.model.generate(
                        text=text,
                        max_tokens=extra.get("max_tokens", 3072),
                        cfg_scale=extra.get("cfg_scale", 3.0),
                        temperature=extra.get("temperature", 1.2),
                        top_p=extra.get("top_p", 0.95),
                        cfg_filter_top_k=45,
                        verbose=False,
                    )
                    if isinstance(output, list):
                        output = output[0]
                    audio_np = np.asarray(output, dtype=np.float32)
                    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                        peak = max(abs(audio_np.max()), abs(audio_np.min()))
                        if peak > 0:
                            audio_np = audio_np / peak
                    sr = svc.SAMPLE_RATE

                from base_service import wav_bytes_from_numpy
                wav_data = wav_bytes_from_numpy(audio_np, sr)
                elapsed = time.time() - start_time

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
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
    print("Dia 1.6B TTS service ready")
    print("Deploy with: modal deploy modal/dia_service.py")
