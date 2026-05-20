"""
Sesame CSM 1B TTS Service - Conversational Speech Model.

Capabilities:
- 1B params, Llama backbone + Mimi audio codec
- Conversation-context-aware generation (prior turns improve quality)
- Multiple speaker IDs (integer-indexed)
- Generates RVQ audio codes, decoded to 24kHz audio
- Watermarked output
- Apache 2.0 licensed

Deploy: modal deploy modal/sesame_csm_service.py
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

app = create_modal_app("sesame-csm")
voice_cache = create_voice_volume("sesame-csm")
model_cache = create_model_volume()

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.10"))
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "moshi",
    )
    .run_commands(
        "git clone https://github.com/SesameAILabs/csm.git /opt/csm",
        "cd /opt/csm && pip install -r requirements.txt",
    )
    .env({"NO_TORCH_COMPILE": "1", "PYTHONPATH": "/opt/csm"})
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
class SesameCsmService(ModalTTSService):
    """Sesame CSM 1B — conversational speech generation."""

    MODEL_NAME = "sesame-csm"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 24000
    VOICE_REF_SR = 24000
    VOICE_REF_MAX_DUR = 30
    FEATURES = [
        "conversational_context", "multi_speaker", "voice_cloning",
    ]

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        import sys
        sys.path.insert(0, "/opt/csm")

        os.environ["HF_HOME"] = "/model-cache/sesame-csm"

        from generator import load_csm_1b
        self.generator = load_csm_1b(device="cuda")
        self._sample_rate = self.generator.sample_rate

        # Conversation context cache per voice_id: list of Segment
        self._context_cache: dict[str, list] = {}
        self._max_context_turns = 10

        print(f"Sesame CSM 1B loaded — sample_rate={self._sample_rate}")

    def synthesize_impl(self, text: str, voice_id: str, **kwargs):
        """Synthesize speech with optional conversation context.

        Extra kwargs:
            speaker (int): Speaker ID (default 0)
            max_audio_length_ms (float): Max generation length in ms (default 10000)
            temperature (float): Sampling temperature (default 0.9)
            topk (int): Top-k sampling (default 50)
            use_context (bool): Use stored conversation context (default True)
            context_audio_paths (list[dict]): Explicit context segments, each with
                keys: speaker (int), text (str), audio_path (str)
        """
        import torch
        import torchaudio
        import sys
        sys.path.insert(0, "/opt/csm")
        from generator import Segment

        speaker = kwargs.get("speaker", 0)
        max_audio_length_ms = kwargs.get("max_audio_length_ms", 10_000)
        temperature = kwargs.get("temperature", 0.9)
        topk = kwargs.get("topk", 50)
        use_context = kwargs.get("use_context", True)

        # Build context segments
        context: list[Segment] = []

        if use_context and voice_id in self._context_cache:
            context = self._context_cache[voice_id][-self._max_context_turns:]

        # Explicit context segments override cached context
        explicit_ctx = kwargs.get("context_audio_paths")
        if explicit_ctx:
            context = []
            for seg in explicit_ctx:
                audio_path = seg.get("audio_path", "")
                if not os.path.exists(audio_path):
                    continue
                audio_tensor, sr = torchaudio.load(audio_path)
                if sr != self._sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor.squeeze(0), sr, self._sample_rate,
                    )
                else:
                    audio_tensor = audio_tensor.squeeze(0)
                context.append(Segment(
                    speaker=seg.get("speaker", 0),
                    text=seg.get("text", ""),
                    audio=audio_tensor,
                ))

        # Voice reference as single-turn context if no other context
        if not context and self.voice_exists(voice_id):
            ref_path = self.get_voice_ref_path(voice_id)
            if os.path.exists(ref_path):
                audio_tensor, sr = torchaudio.load(ref_path)
                if sr != self._sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor.squeeze(0), sr, self._sample_rate,
                    )
                else:
                    audio_tensor = audio_tensor.squeeze(0)
                context = [Segment(
                    speaker=speaker,
                    text="",
                    audio=audio_tensor,
                )]

        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )

        audio_np = audio.cpu().float().numpy()

        # Cache this turn for future context
        if use_context:
            if voice_id not in self._context_cache:
                self._context_cache[voice_id] = []
            self._context_cache[voice_id].append(Segment(
                speaker=speaker,
                text=text,
                audio=audio.cpu(),
            ))
            # Trim to max context size
            if len(self._context_cache[voice_id]) > self._max_context_turns:
                self._context_cache[voice_id] = self._context_cache[voice_id][
                    -self._max_context_turns:
                ]

        return audio_np, self._sample_rate

    def synthesize_with_url_impl(self, text: str, ref_audio_path: str, **kwargs):
        """One-shot synthesis using ref audio as conversation context."""
        import torch
        import torchaudio
        import sys
        sys.path.insert(0, "/opt/csm")
        from generator import Segment

        speaker = kwargs.get("speaker", 0)
        max_audio_length_ms = kwargs.get("max_audio_length_ms", 10_000)
        temperature = kwargs.get("temperature", 0.9)
        topk = kwargs.get("topk", 50)

        audio_tensor, sr = torchaudio.load(ref_audio_path)
        if sr != self._sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), sr, self._sample_rate,
            )
        else:
            audio_tensor = audio_tensor.squeeze(0)

        context = [Segment(
            speaker=speaker,
            text="",
            audio=audio_tensor,
        )]

        audio = self.generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )

        return audio.cpu().float().numpy(), self._sample_rate

    def health_extra(self):
        return {
            "version": "csm-1b",
            "actual_sample_rate": self._sample_rate,
            "max_context_turns": self._max_context_turns,
            "cached_contexts": list(self._context_cache.keys()),
        }

    def extra_routes(self, api):
        from fastapi import Request
        from fastapi.responses import Response, JSONResponse
        import time

        svc = self

        @api.post("/synthesize_conversation")
        async def synthesize_conversation(request: Request):
            """Synthesize with explicit multi-turn conversation context.

            Body:
                text: Text to generate speech for
                voice_id: Voice reference ID
                speaker: Speaker ID (int, default 0)
                context: List of prior turns, each with:
                    - speaker (int)
                    - text (str)
                    - audio_base64 (str): base64-encoded WAV audio
                max_audio_length_ms: Max generation length (default 10000)
                temperature: Sampling temperature (default 0.9)
                topk: Top-k sampling (default 50)
            """
            start_time = time.time()
            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            text = body.get("text", "")
            voice_id = body.get("voice_id", "_conversation")
            speaker = body.get("speaker", 0)

            if not text:
                return JSONResponse(status_code=400, content={"error": "text is required"})

            context_data = body.get("context", [])

            try:
                import sys
                sys.path.insert(0, "/opt/csm")
                from generator import Segment
                import torch
                import torchaudio
                import base64
                import io

                context = []
                for turn in context_data:
                    audio_b64 = turn.get("audio_base64")
                    if not audio_b64:
                        continue
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))
                    if sr != svc._sample_rate:
                        audio_tensor = torchaudio.functional.resample(
                            audio_tensor.squeeze(0), sr, svc._sample_rate,
                        )
                    else:
                        audio_tensor = audio_tensor.squeeze(0)
                    context.append(Segment(
                        speaker=turn.get("speaker", 0),
                        text=turn.get("text", ""),
                        audio=audio_tensor.to("cuda"),
                    ))

                audio = svc.generator.generate(
                    text=text,
                    speaker=speaker,
                    context=context,
                    max_audio_length_ms=body.get("max_audio_length_ms", 10_000),
                    temperature=body.get("temperature", 0.9),
                    topk=body.get("topk", 50),
                )

                from base_service import wav_bytes_from_numpy
                audio_np = audio.cpu().float().numpy()
                wav_data = wav_bytes_from_numpy(audio_np, svc._sample_rate)
                elapsed = time.time() - start_time

                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Sample-Rate": str(svc._sample_rate),
                        "X-Context-Turns": str(len(context)),
                    },
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.post("/clear_context")
        async def clear_context(request: Request):
            """Clear cached conversation context for a voice_id."""
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

            voice_id = body.get("voice_id")
            if not voice_id:
                return JSONResponse(status_code=400, content={"error": "voice_id is required"})

            if voice_id in svc._context_cache:
                turns_cleared = len(svc._context_cache[voice_id])
                del svc._context_cache[voice_id]
                return {"status": "cleared", "voice_id": voice_id, "turns_cleared": turns_cleared}
            return {"status": "no_context", "voice_id": voice_id}

    @modal.asgi_app()
    def web_app(self):
        return self.build_app()


@app.local_entrypoint()
def main():
    print("Sesame CSM 1B TTS service ready")
    print("Deploy with: modal deploy modal/sesame_csm_service.py")
