"""
VibeVoice Realtime 0.5B TTS Service - Microsoft's streaming real-time TTS.

Capabilities:
- 0.5B params, ~300ms first-audible latency
- Streaming text input, 24kHz output
- 7 built-in English speakers (Carter, Davis, Emma, Frank, Grace, Mike, Samuel)
- 11 experimental English style voices + 9 multilingual voices
- ~10 min max generation length per call
- MIT licensed

Deploy: modal deploy modal/vibevoice_service.py
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

app = create_modal_app("vibevoice-realtime")
voice_cache = create_voice_volume("vibevoice")
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
        "transformers",
        "accelerate",
        "flash-attn",
    )
    .run_commands(
        "pip install git+https://github.com/microsoft/VibeVoice.git[streamingtts]"
    )
)

# Built-in speaker presets bundled with the model
BUILTIN_SPEAKERS = [
    "Carter", "Davis", "Emma", "Frank", "Grace", "Mike", "Samuel",
]


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
class VibeVoiceRealtimeService(ModalTTSService):
    """VibeVoice Realtime 0.5B — streaming text-to-speech."""

    MODEL_NAME = "vibevoice-realtime"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 24000
    VOICE_REF_SR = 24000
    VOICE_REF_MAX_DUR = 30
    FEATURES = ["preset_voices", "streaming", "realtime"]

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        import torch
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        model_id = "microsoft/VibeVoice-Realtime-0.5B"
        cache_dir = "/model-cache/vibevoice-realtime"

        self.processor = VibeVoiceStreamingProcessor.from_pretrained(
            model_id, cache_dir=cache_dir,
        )
        self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir,
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)

        # Load built-in voice presets (.pt files shipped with the model)
        self._voice_presets: dict[str, object] = {}
        voices_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "demo", "voices", "streaming_model",
        )
        # Try HF cache location as fallback
        if not os.path.isdir(voices_dir):
            import glob
            pattern = f"{cache_dir}/**/demo/voices/streaming_model"
            matches = glob.glob(pattern, recursive=True)
            if matches:
                voices_dir = matches[0]

        if os.path.isdir(voices_dir):
            for name in BUILTIN_SPEAKERS:
                pt_path = os.path.join(voices_dir, f"{name}.pt")
                if os.path.exists(pt_path):
                    self._voice_presets[name.lower()] = torch.load(
                        pt_path, map_location="cuda", weights_only=False,
                    )
                    print(f"Loaded voice preset: {name}")

        print(f"VibeVoice Realtime loaded — {len(self._voice_presets)} presets")

    def _get_voice_preset(self, voice_id: str):
        """Return a voice preset tensor, or None for cloned voices."""
        return self._voice_presets.get(voice_id.lower())

    def synthesize_impl(self, text: str, voice_id: str, **kwargs):
        import torch

        cfg_scale = kwargs.get("cfg_scale", 1.5)

        # Check built-in presets first, then cloned voice references
        preset = self._get_voice_preset(voice_id)
        if preset is None and not self.voice_exists(voice_id):
            raise ValueError(f"Voice '{voice_id}' not found")

        if preset is not None:
            # Built-in preset: use cached prompt
            inputs = self.processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=preset,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
        else:
            # Cloned voice: load reference audio
            ref_path = self.get_voice_ref_path(voice_id)
            import torchaudio
            audio, sr = torchaudio.load(ref_path)
            if sr != self.SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, self.SAMPLE_RATE)
            inputs = self.processor(
                text=text,
                audio=audio.squeeze(0),
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )

        audio_np = outputs.speech_outputs[0].cpu().float().numpy()
        return audio_np, self.SAMPLE_RATE

    def voice_exists(self, voice_id: str) -> bool:
        if voice_id.lower() in self._voice_presets:
            return True
        return super().voice_exists(voice_id)

    def list_voice_ids(self) -> list[str]:
        preset_ids = list(self._voice_presets.keys())
        cloned_ids = super().list_voice_ids()
        return preset_ids + cloned_ids

    def health_extra(self):
        return {
            "builtin_speakers": list(self._voice_presets.keys()),
            "version": "realtime-0.5b",
        }

    def extra_routes(self, api):
        from fastapi.responses import JSONResponse

        svc = self

        @api.get("/speakers")
        def list_speakers():
            """List built-in speaker presets (no cloning needed)."""
            return {
                "speakers": [
                    {"id": sid, "type": "builtin"}
                    for sid in svc._voice_presets.keys()
                ],
            }

    @modal.asgi_app()
    def web_app(self):
        return self.build_app()


@app.local_entrypoint()
def main():
    print("VibeVoice Realtime 0.5B TTS service ready")
    print("Deploy with: modal deploy modal/vibevoice_service.py")
