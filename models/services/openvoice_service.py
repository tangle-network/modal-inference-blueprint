"""
OpenVoice V2 TTS Service - Zero-shot voice cloning with style control.

Capabilities:
- Instant zero-shot voice cloning from short reference audio
- Tone color conversion (any voice -> target voice)
- Built-in base speakers via MeloTTS (EN, ES, FR, ZH, JP, KR)
- Style/emotion transfer
- MIT license

Deploy: modal deploy modal/openvoice_service.py

Requirements:
- MeloTTS for base TTS generation
- OpenVoice for tone color conversion
- A10G GPU (24GB VRAM)
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

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("openvoice")
voice_cache = create_voice_volume("openvoice")

CKPT_DIR = "/openvoice-checkpoints"

# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.10"))
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "librosa==0.9.1",
        "wavmark==0.0.3",
        "pypinyin==0.50.0",
        "cn2an==0.5.22",
        "jieba==0.42.1",
        "eng_to_ipa==0.0.2",
        "inflect==7.0.0",
        "unidecode==1.3.7",
        "whisper-timestamped==1.14.2",
        "langid==1.1.6",
        "pydub==0.25.1",
    )
    .run_commands(
        "pip install git+https://github.com/myshell-ai/OpenVoice.git",
        "pip install git+https://github.com/myshell-ai/MeloTTS.git",
        "python -m unidic download",
    )
    .run_commands(
        # Download OpenVoice V2 checkpoints at build time
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('myshell-ai/OpenVoiceV2', local_dir='/openvoice-checkpoints/v2')"
        "\"",
    )
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=180,
    volumes={"/voice-cache": voice_cache},
    allow_concurrent_inputs=5,
)
class OpenVoiceService(PhonyTTSService):
    """OpenVoice V2 TTS with zero-shot voice cloning."""

    MODEL_NAME = "openvoice-v2"
    GPU_TYPE = "A10G"
    SAMPLE_RATE = 24000
    VOICE_REF_SR = 16000
    VOICE_REF_MAX_DUR = 30
    FEATURES = ["voice_cloning", "tone_color_conversion", "multilingual", "style_control"]

    _voice_cache_volume = voice_cache

    @modal.enter()
    def load_model(self):
        super().load_model()

    def setup_model(self):
        """Load OpenVoice converter and MeloTTS base model."""
        import torch
        from openvoice.api import ToneColorConverter
        from melo.api import TTS as MeloTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Tone color converter
        self.converter = ToneColorConverter(
            f"{CKPT_DIR}/v2/converter/config.json", device=device,
        )
        self.converter.load_ckpt(f"{CKPT_DIR}/v2/converter/checkpoint.pth")

        # MeloTTS base model (English default, others loaded on demand)
        self.melo_models = {}
        self.melo_models["EN"] = MeloTTS(language="EN", device=device)

        # Cache for speaker embeddings
        self._se_cache = {}

        print("OpenVoice V2 converter + MeloTTS EN loaded.")

    def _get_melo(self, language: str):
        """Get or lazily load a MeloTTS model for a language."""
        lang = language.upper()
        if lang not in self.melo_models:
            from melo.api import TTS as MeloTTS
            self.melo_models[lang] = MeloTTS(language=lang, device=self.device)
            print(f"Loaded MeloTTS for {lang}")
        return self.melo_models[lang]

    def _get_target_se(self, voice_id: str):
        """Extract or retrieve cached speaker embedding for a voice."""
        if voice_id in self._se_cache:
            return self._se_cache[voice_id]

        from openvoice import se_extractor

        ref_path = self.get_voice_ref_path(voice_id)
        target_se, _ = se_extractor.get_se(
            ref_path, self.converter, target_dir="/tmp/openvoice_se", vad=True,
        )
        self._se_cache[voice_id] = target_se

        # Evict oldest if cache too large
        if len(self._se_cache) > 20:
            oldest = next(iter(self._se_cache))
            del self._se_cache[oldest]

        return target_se

    def synthesize_impl(self, text: str, voice_id: str, **kwargs) -> tuple:
        import numpy as np
        import soundfile as sf
        import tempfile

        language = kwargs.get("language", "EN")
        speed = kwargs.get("speed", 1.0)

        melo = self._get_melo(language)
        speaker_ids = melo.hps.data.spk2id

        # Use first available speaker from MeloTTS as source
        src_speaker = list(speaker_ids.keys())[0]

        # Generate base TTS audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            src_path = f.name
        melo.tts_to_file(text, speaker_ids[src_speaker], src_path, speed=speed)

        # Extract source speaker embedding
        from openvoice import se_extractor
        source_se, _ = se_extractor.get_se(
            src_path, self.converter, target_dir="/tmp/openvoice_src_se", vad=False,
        )

        # Extract target speaker embedding from cloned voice
        target_se = self._get_target_se(voice_id)

        # Convert tone color
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        self.converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            message="@ph0ny",
        )

        audio, sr = sf.read(out_path, dtype="float32")

        # Cleanup temp files
        for p in (src_path, out_path):
            if os.path.exists(p):
                os.unlink(p)

        # Resample if needed
        if sr != self.SAMPLE_RATE:
            import torchaudio
            import torch
            tensor = torch.from_numpy(audio).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, sr, self.SAMPLE_RATE)
            audio = tensor.squeeze().numpy()

        return audio, self.SAMPLE_RATE

    def synthesize_with_url_impl(self, text: str, ref_audio_path: str, **kwargs) -> tuple:
        """One-shot synthesis using a temp reference audio directly."""
        import numpy as np
        import soundfile as sf
        import tempfile
        from openvoice import se_extractor

        language = kwargs.get("language", "EN")
        speed = kwargs.get("speed", 1.0)

        melo = self._get_melo(language)
        speaker_ids = melo.hps.data.spk2id
        src_speaker = list(speaker_ids.keys())[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            src_path = f.name
        melo.tts_to_file(text, speaker_ids[src_speaker], src_path, speed=speed)

        source_se, _ = se_extractor.get_se(
            src_path, self.converter, target_dir="/tmp/openvoice_src_se", vad=False,
        )
        target_se, _ = se_extractor.get_se(
            ref_audio_path, self.converter, target_dir="/tmp/openvoice_tgt_se", vad=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        self.converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            message="@ph0ny",
        )

        audio, sr = sf.read(out_path, dtype="float32")

        for p in (src_path, out_path):
            if os.path.exists(p):
                os.unlink(p)

        if sr != self.SAMPLE_RATE:
            import torchaudio
            import torch
            tensor = torch.from_numpy(audio).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, sr, self.SAMPLE_RATE)
            audio = tensor.squeeze().numpy()

        return audio, self.SAMPLE_RATE

    def health_extra(self):
        return {
            "loaded_languages": list(self.melo_models.keys()),
            "available_languages": ["EN", "ES", "FR", "ZH", "JP", "KR"],
            "cached_voices": len(self._se_cache),
        }

    def extra_routes(self, api):
        """Add OpenVoice-specific endpoints."""
        from fastapi.responses import JSONResponse
        svc = self

        @api.get("/languages")
        def list_languages():
            return {
                "loaded": list(svc.melo_models.keys()),
                "available": ["EN", "ES", "FR", "ZH", "JP", "KR"],
            }

    @modal.asgi_app()
    def web_app(self):
        return self.build_app()


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("OpenVoice V2 TTS Service ready")
    print("Deploy with: modal deploy modal/openvoice_service.py")
