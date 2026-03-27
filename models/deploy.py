"""
Modal deployment orchestrator — reads registry.toml and deploys models as Modal apps.

Usage:
    # List all registered models
    python models/deploy.py list

    # Deploy a single model
    python models/deploy.py deploy --model llama-4-maverick

    # Deploy all models of a task type
    python models/deploy.py deploy --task text-generation

    # Deploy all models
    python models/deploy.py deploy --all

    # Dry run (print Modal app configs without deploying)
    python models/deploy.py deploy --model flux-1-dev --dry-run

    # Generate operator.toml entries for deployed models
    python models/deploy.py gen-config --org your-modal-org

    # Health check deployed models
    python models/deploy.py health --org your-modal-org
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


REGISTRY_PATH = Path(__file__).parent / "registry.toml"

# Modal GPU class mapping
GPU_MAP = {
    "T4": "T4",
    "A10G": "A10G",
    "L4": "L4",
    "L40S": "L40S",
    "A100-40GB": "A100",
    "A100-80GB": "A100-80GB",
    "H100": "H100",
}

# Container idle timeout by task type (seconds)
IDLE_TIMEOUT = {
    "text-generation": 300,
    "image-generation": 180,
    "video-generation": 120,
    "video-avatar": 180,
    "video-lipsync": 180,
    "video-stitch": 120,
    "video-understanding": 300,
    "tts": 300,
    "stt": 300,
    "music-generation": 120,
    "embedding": 600,
    "rerank": 600,
    "diarize": 300,
    "vad": 300,
}

# Max concurrent inputs by task type
CONCURRENCY = {
    "text-generation": 32,
    "image-generation": 4,
    "video-generation": 1,
    "video-avatar": 2,
    "video-lipsync": 2,
    "video-stitch": 10,
    "video-understanding": 8,
    "tts": 8,
    "stt": 16,
    "music-generation": 2,
    "embedding": 64,
    "rerank": 64,
    "diarize": 4,
    "vad": 16,
}


@dataclass
class ModelSpec:
    model_id: str
    name: str
    task_type: str
    gpu_type: str
    gpu_count: int
    vram_required_mib: int
    recommended_tp: int
    modal_image: str
    inference_engine: str
    endpoint_path: str
    context_length: int = 0
    estimated_latency_ms: int = 0
    price_per_m_tokens: float = 0.0
    price_per_image: float = 0.0
    price_per_second: float = 0.0
    notes: str = ""


def load_registry(path: Path = REGISTRY_PATH) -> list[ModelSpec]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return [ModelSpec(**m) for m in data.get("models", [])]


def filter_models(
    models: list[ModelSpec],
    name: Optional[str] = None,
    task: Optional[str] = None,
    gpu: Optional[str] = None,
) -> list[ModelSpec]:
    result = models
    if name:
        result = [m for m in result if m.name == name]
    if task:
        result = [m for m in result if m.task_type == task]
    if gpu:
        result = [m for m in result if m.gpu_type == gpu]
    return result


# ---------------------------------------------------------------------------
# Modal app generation per inference engine
# ---------------------------------------------------------------------------

def _vllm_app_source(spec: ModelSpec, org: str) -> str:
    """Generate a Modal app module for vLLM-served models."""
    app_name = f"inference-{spec.name}"
    gpu_class = GPU_MAP[spec.gpu_type]
    gpu_spec = f'"{gpu_class}"' if spec.gpu_count == 1 else f'modal.gpu.{gpu_class}(count={spec.gpu_count})'
    idle = IDLE_TIMEOUT.get(spec.task_type, 300)
    concurrency = CONCURRENCY.get(spec.task_type, 16)
    tp = spec.recommended_tp

    extra_args = ""
    if spec.context_length > 32768:
        extra_args += f'\n        "--max-model-len", "{min(spec.context_length, 131072)}",'
    if tp > 1:
        extra_args += f'\n        "--tensor-parallel-size", "{tp}",'
    # MoE expert parallel for large MoE models
    if "expert" in spec.notes.lower() or "moe" in spec.notes.lower():
        extra_args += '\n        "--enable-expert-parallel",'
    extra_args += '\n        "--dtype", "auto",'
    extra_args += '\n        "--enforce-eager", "false",'

    return f'''"""Auto-generated Modal app for {spec.name} ({spec.model_id})."""
import modal

app = modal.App("{app_name}")

MODEL_ID = "{spec.model_id}"

vllm_image = (
    modal.Image.from_registry("{spec.modal_image}", add_python="3.11")
    .pip_install("huggingface-hub[hf_transfer]>=0.25.0")
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

MINUTES = 60

@app.function(
    image=vllm_image,
    gpu={gpu_spec},
    timeout=10 * MINUTES,
    container_idle_timeout={idle},
    allow_concurrent_inputs={concurrency},
    volumes={{
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache-{spec.name}", create_if_missing=True
        ),
    }},
)
@modal.asgi_app()
def serve():
    import subprocess, os
    # vLLM OpenAI-compatible server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", "8000",{extra_args}
    ]
    proc = subprocess.Popen(cmd)
    # Wait for server to be ready
    import time, urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen("http://localhost:8000/health")
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("vLLM server failed to start")

    # Proxy via ASGI
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.middleware import Middleware
    import httpx
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    async def proxy(request: Request):
        url = f"http://localhost:8000{{request.url.path}}"
        client = httpx.AsyncClient()
        body = await request.body()
        resp = await client.request(
            request.method, url,
            content=body,
            headers=dict(request.headers),
        )
        return StreamingResponse(
            iter([resp.content]),
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    from starlette.routing import Route
    return Starlette(routes=[
        Route("/health", lambda r: __import__("starlette.responses", fromlist=["JSONResponse"]).JSONResponse({{"status": "ok", "model": MODEL_ID}}), methods=["GET"]),
        Route("/v1/{{path:path}}", proxy, methods=["GET", "POST"]),
        Route("/{{path:path}}", proxy, methods=["GET", "POST"]),
    ])
'''


def _diffusers_image_app_source(spec: ModelSpec, org: str) -> str:
    """Generate a Modal app for diffusers-based image generation."""
    app_name = f"inference-{spec.name}"
    gpu_class = GPU_MAP[spec.gpu_type]
    idle = IDLE_TIMEOUT.get(spec.task_type, 180)
    concurrency = CONCURRENCY.get(spec.task_type, 4)

    # Determine pipeline class based on model
    if "flux" in spec.model_id.lower():
        pipeline_cls = "FluxPipeline"
        pipeline_import = "from diffusers import FluxPipeline"
    elif "sd3" in spec.name or "stable-diffusion-3" in spec.model_id.lower():
        pipeline_cls = "StableDiffusion3Pipeline"
        pipeline_import = "from diffusers import StableDiffusion3Pipeline"
    elif "sdxl" in spec.name:
        pipeline_cls = "StableDiffusionXLPipeline"
        pipeline_import = "from diffusers import StableDiffusionXLPipeline"
    elif "pixart" in spec.name:
        pipeline_cls = "PixArtSigmaPipeline"
        pipeline_import = "from diffusers import PixArtSigmaPipeline"
    elif "kolors" in spec.name:
        pipeline_cls = "KolorsPipeline"
        pipeline_import = "from diffusers import KolorsPipeline"
    elif "hunyuan" in spec.name and "video" not in spec.name:
        pipeline_cls = "HunyuanDiTPipeline"
        pipeline_import = "from diffusers import HunyuanDiTPipeline"
    else:
        pipeline_cls = "AutoPipelineForText2Image"
        pipeline_import = "from diffusers import AutoPipelineForText2Image"

    return f'''"""Auto-generated Modal app for {spec.name} ({spec.model_id})."""
import modal
import io

app = modal.App("{app_name}")

MODEL_ID = "{spec.model_id}"

image = (
    modal.Image.from_registry("{spec.modal_image}")
    .pip_install(
        "torch==2.5.1", "diffusers>=0.31.0", "transformers>=4.44.0",
        "accelerate>=1.0.0", "safetensors>=0.4.0", "sentencepiece",
        "protobuf", "huggingface-hub[hf_transfer]>=0.25.0", "Pillow",
        "fastapi", "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

MINUTES = 60

@app.cls(
    image=image,
    gpu="{gpu_class}",
    timeout=5 * MINUTES,
    container_idle_timeout={idle},
    allow_concurrent_inputs={concurrency},
    volumes={{
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache-{spec.name}", create_if_missing=True
        ),
    }},
)
class Inference:
    @modal.enter()
    def load(self):
        import torch
        {pipeline_import}
        self.pipe = {pipeline_cls}.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir="/root/.cache/huggingface",
        ).to("cuda")

    @modal.web_endpoint(method="POST")
    def generate(self, request: dict):
        import torch, base64
        from fastapi.responses import JSONResponse
        from PIL import Image

        prompt = request.get("prompt", "")
        negative_prompt = request.get("negative_prompt", "")
        width = request.get("width", 1024)
        height = request.get("height", 1024)
        steps = request.get("num_inference_steps", 28)
        guidance = request.get("guidance_scale", 7.0)
        seed = request.get("seed")
        n = request.get("n", 1)

        generator = torch.Generator("cuda").manual_seed(seed) if seed else None

        images = []
        for _ in range(n):
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            img = result.images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode())

        # OpenAI-compatible response
        return JSONResponse({{
            "created": __import__("time").time(),
            "data": [{{"b64_json": b}} for b in images],
        }})

    @modal.web_endpoint(method="GET")
    def health(self):
        return {{"status": "ok", "model": MODEL_ID}}
'''


def _diffusers_video_app_source(spec: ModelSpec, org: str) -> str:
    """Generate a Modal app for diffusers-based video generation."""
    app_name = f"inference-{spec.name}"
    gpu_class = GPU_MAP[spec.gpu_type]
    gpu_spec = f'"{gpu_class}"' if spec.gpu_count == 1 else f'modal.gpu.{gpu_class}(count={spec.gpu_count})'
    idle = IDLE_TIMEOUT.get(spec.task_type, 120)

    if "wan" in spec.model_id.lower():
        pipeline_cls = "WanPipeline"
        pipeline_import = "from diffusers import WanPipeline"
    elif "cogvideo" in spec.model_id.lower():
        pipeline_cls = "CogVideoXPipeline"
        pipeline_import = "from diffusers import CogVideoXPipeline"
    elif "ltx" in spec.model_id.lower():
        pipeline_cls = "LTXPipeline"
        pipeline_import = "from diffusers import LTXPipeline"
    elif "hunyuan" in spec.model_id.lower():
        pipeline_cls = "HunyuanVideoPipeline"
        pipeline_import = "from diffusers import HunyuanVideoPipeline"
    else:
        pipeline_cls = "DiffusionPipeline"
        pipeline_import = "from diffusers import DiffusionPipeline"

    return f'''"""Auto-generated Modal app for {spec.name} ({spec.model_id})."""
import modal
import io

app = modal.App("{app_name}")

MODEL_ID = "{spec.model_id}"

image = (
    modal.Image.from_registry("{spec.modal_image}")
    .pip_install(
        "torch==2.5.1", "diffusers>=0.31.0", "transformers>=4.44.0",
        "accelerate>=1.0.0", "safetensors>=0.4.0", "sentencepiece",
        "imageio[ffmpeg]", "imageio", "numpy",
        "huggingface-hub[hf_transfer]>=0.25.0",
        "fastapi", "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

MINUTES = 60

@app.cls(
    image=image,
    gpu={gpu_spec},
    timeout=15 * MINUTES,
    container_idle_timeout={idle},
    allow_concurrent_inputs=1,
    volumes={{
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache-{spec.name}", create_if_missing=True
        ),
    }},
)
class Inference:
    @modal.enter()
    def load(self):
        import torch
        {pipeline_import}
        self.pipe = {pipeline_cls}.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir="/root/.cache/huggingface",
        ).to("cuda")

    @modal.web_endpoint(method="POST")
    def generate(self, request: dict):
        import torch, base64, tempfile
        from fastapi.responses import JSONResponse
        import imageio

        prompt = request.get("prompt", "")
        num_frames = request.get("num_frames", 49)
        fps = request.get("fps", 16)
        width = request.get("width", 480)
        height = request.get("height", 720)
        steps = request.get("num_inference_steps", 50)
        seed = request.get("seed")

        generator = torch.Generator("cuda").manual_seed(seed) if seed else None

        result = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
        )
        frames = result.frames[0]  # list of PIL images or tensor

        # Encode as mp4
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            import numpy as np
            frame_arrays = [np.array(frame) for frame in frames]
            imageio.mimwrite(f.name, frame_arrays, fps=fps, codec="libx264")
            f.seek(0)
            video_bytes = open(f.name, "rb").read()

        return JSONResponse({{
            "created": __import__("time").time(),
            "data": [{{"b64_json": base64.b64encode(video_bytes).decode(), "content_type": "video/mp4"}}],
        }})

    @modal.web_endpoint(method="GET")
    def health(self):
        return {{"status": "ok", "model": MODEL_ID}}
'''


def _tts_app_source(spec: ModelSpec, org: str) -> str:
    """Generate a Modal app for TTS models."""
    app_name = f"inference-{spec.name}"
    gpu_class = GPU_MAP[spec.gpu_type]
    idle = IDLE_TIMEOUT.get(spec.task_type, 300)
    concurrency = CONCURRENCY.get(spec.task_type, 8)

    if "kokoro" in spec.name:
        engine_code = '''
    @modal.enter()
    def load(self):
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code="a")

    @modal.web_endpoint(method="POST")
    def synthesize(self, request: dict):
        import io, base64, soundfile as sf
        from fastapi.responses import Response
        text = request.get("text", request.get("input", ""))
        voice = request.get("voice", "af_heart")
        fmt = request.get("response_format", "wav")
        audio_chunks = []
        for _, _, audio in self.pipeline(text, voice=voice):
            audio_chunks.append(audio)
        import numpy as np
        audio = np.concatenate(audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format=fmt.upper())
        buf.seek(0)
        return Response(content=buf.read(), media_type=f"audio/{fmt}")'''
        extra_pip = '"kokoro>=0.9.0", "soundfile", "numpy",'
    elif "f5" in spec.name:
        engine_code = '''
    @modal.enter()
    def load(self):
        from f5_tts.api import F5TTS
        self.model = F5TTS()

    @modal.web_endpoint(method="POST")
    def synthesize(self, request: dict):
        import io, soundfile as sf
        from fastapi.responses import Response
        text = request.get("text", request.get("input", ""))
        ref_audio = request.get("ref_audio_path", None)
        ref_text = request.get("ref_text", "")
        audio, sr, _ = self.model.infer(ref_file=ref_audio or "", ref_text=ref_text, gen_text=text)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")'''
        extra_pip = '"f5-tts>=0.4.0", "soundfile", "numpy",'
    elif "parler" in spec.name:
        engine_code = '''
    @modal.enter()
    def load(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    @modal.web_endpoint(method="POST")
    def synthesize(self, request: dict):
        import io, soundfile as sf
        from fastapi.responses import Response
        text = request.get("text", request.get("input", ""))
        description = request.get("voice", "A female speaker with a warm, clear voice delivers the text at a moderate pace.")
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to("cuda")
        prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        audio = generation.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio, self.model.config.sampling_rate, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")'''
        extra_pip = '"parler-tts>=0.2.0", "soundfile", "numpy",'
    else:
        # Generic transformers TTS
        engine_code = '''
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModel
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")

    @modal.web_endpoint(method="POST")
    def synthesize(self, request: dict):
        import io, soundfile as sf
        from fastapi.responses import Response
        text = request.get("text", request.get("input", ""))
        inputs = self.processor(text=text, return_tensors="pt").to("cuda")
        audio = self.model.generate(**inputs)
        audio_np = audio.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio_np, 24000, format="WAV")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")'''
        extra_pip = '"soundfile", "numpy",'

    return f'''"""Auto-generated Modal app for {spec.name} ({spec.model_id})."""
import modal

app = modal.App("{app_name}")

MODEL_ID = "{spec.model_id}"

image = (
    modal.Image.from_registry("{spec.modal_image}")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1", "transformers>=4.44.0",
        "accelerate>=1.0.0", "safetensors>=0.4.0",
        {extra_pip}
        "huggingface-hub[hf_transfer]>=0.25.0",
        "fastapi", "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

MINUTES = 60

@app.cls(
    image=image,
    gpu="{gpu_class}",
    timeout=5 * MINUTES,
    container_idle_timeout={idle},
    allow_concurrent_inputs={concurrency},
    volumes={{
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache-{spec.name}", create_if_missing=True
        ),
    }},
)
class Inference:
{engine_code}

    @modal.web_endpoint(method="GET")
    def health(self):
        return {{"status": "ok", "model": MODEL_ID}}
'''


def _stt_app_source(spec: ModelSpec, org: str) -> str:
    """Generate a Modal app for STT models."""
    app_name = f"inference-{spec.name}"
    gpu_class = GPU_MAP[spec.gpu_type]
    idle = IDLE_TIMEOUT.get(spec.task_type, 300)
    concurrency = CONCURRENCY.get(spec.task_type, 16)

    if "faster-whisper" in spec.model_id.lower():
        engine_code = '''
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel("large-v3", device="cuda", compute_type="int8")

    @modal.web_endpoint(method="POST")
    def transcribe(self, file: bytes):
        import io, json, tempfile
        from fastapi.responses import JSONResponse
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(file)
            f.flush()
            segments, info = self.model.transcribe(f.name, beam_size=5)
        text = " ".join(s.text for s in segments)
        return JSONResponse({"text": text, "language": info.language})'''
        extra_pip = '"faster-whisper>=1.0.0",'
    elif "canary" in spec.name:
        engine_code = '''
    @modal.enter()
    def load(self):
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b").to("cuda")

    @modal.web_endpoint(method="POST")
    def transcribe(self, file: bytes):
        import tempfile
        from fastapi.responses import JSONResponse
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(file)
            f.flush()
            result = self.model.transcribe([f.name])
        return JSONResponse({"text": result[0] if result else ""})'''
        extra_pip = '"nemo_toolkit[asr]>=2.0.0",'
    else:
        # Standard Whisper
        engine_code = '''
    @modal.enter()
    def load(self):
        import whisper
        self.model = whisper.load_model("large-v3-turbo" if "turbo" in MODEL_ID else "large-v3", device="cuda")

    @modal.web_endpoint(method="POST")
    def transcribe(self, file: bytes):
        import tempfile
        from fastapi.responses import JSONResponse
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(file)
            f.flush()
            result = self.model.transcribe(f.name)
        return JSONResponse({"text": result["text"], "language": result.get("language", "")})'''
        extra_pip = '"openai-whisper>=20240930",'

    return f'''"""Auto-generated Modal app for {spec.name} ({spec.model_id})."""
import modal

app = modal.App("{app_name}")

MODEL_ID = "{spec.model_id}"

image = (
    modal.Image.from_registry("{spec.modal_image}")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        {extra_pip}
        "huggingface-hub[hf_transfer]>=0.25.0",
        "fastapi", "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

MINUTES = 60

@app.cls(
    image=image,
    gpu="{gpu_class}",
    timeout=5 * MINUTES,
    container_idle_timeout={idle},
    allow_concurrent_inputs={concurrency},
    volumes={{
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache-{spec.name}", create_if_missing=True
        ),
    }},
)
class Inference:
{engine_code}

    @modal.web_endpoint(method="GET")
    def health(self):
        return {{"status": "ok", "model": MODEL_ID}}
'''


def _music_app_source(spec: ModelSpec, org: str) -> str:
    """Generate a Modal app for music generation."""
    app_name = f"inference-{spec.name}"
    gpu_class = GPU_MAP[spec.gpu_type]
    idle = IDLE_TIMEOUT.get(spec.task_type, 120)

    if "stable-audio" in spec.model_id.lower():
        load_code = '''
        from diffusers import StableAudioPipeline
        self.pipe = StableAudioPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")'''
        gen_code = '''
        audio = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            audio_end_in_s=duration,
            generator=generator,
        ).audios[0]
        sr = 44100'''
    else:
        load_code = '''
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")'''
        gen_code = '''
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
        max_tokens = int(duration * 50)  # ~50 tokens/sec for musicgen
        audio_values = self.model.generate(**inputs, max_new_tokens=max_tokens)
        audio = audio_values[0, 0].cpu().numpy()
        sr = self.model.config.audio_encoder.sampling_rate'''

    return f'''"""Auto-generated Modal app for {spec.name} ({spec.model_id})."""
import modal

app = modal.App("{app_name}")

MODEL_ID = "{spec.model_id}"

image = (
    modal.Image.from_registry("{spec.modal_image}")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1", "transformers>=4.44.0",
        "diffusers>=0.31.0", "accelerate>=1.0.0", "safetensors>=0.4.0",
        "soundfile", "numpy",
        "huggingface-hub[hf_transfer]>=0.25.0",
        "fastapi", "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

MINUTES = 60

@app.cls(
    image=image,
    gpu="{gpu_class}",
    timeout=10 * MINUTES,
    container_idle_timeout={idle},
    allow_concurrent_inputs=2,
    volumes={{
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache-{spec.name}", create_if_missing=True
        ),
    }},
)
class Inference:
    @modal.enter()
    def load(self):
        import torch
{load_code}

    @modal.web_endpoint(method="POST")
    def generate(self, request: dict):
        import io, base64, soundfile as sf, torch
        from fastapi.responses import JSONResponse

        prompt = request.get("prompt", "")
        duration = request.get("duration", 10.0)
        steps = request.get("num_inference_steps", 100)
        seed = request.get("seed")
        generator = torch.Generator("cuda").manual_seed(seed) if seed else None
{gen_code}

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        return JSONResponse({{
            "created": __import__("time").time(),
            "data": [{{"b64_json": base64.b64encode(buf.read()).decode(), "content_type": "audio/wav"}}],
        }})

    @modal.web_endpoint(method="GET")
    def health(self):
        return {{"status": "ok", "model": MODEL_ID}}
'''


def _generic_app_source(spec: ModelSpec, org: str) -> str:
    """Fallback: generic transformers-based Modal app."""
    app_name = f"inference-{spec.name}"
    gpu_class = GPU_MAP[spec.gpu_type]
    idle = IDLE_TIMEOUT.get(spec.task_type, 300)

    return f'''"""Auto-generated Modal app for {spec.name} ({spec.model_id})."""
import modal

app = modal.App("{app_name}")

MODEL_ID = "{spec.model_id}"

image = (
    modal.Image.from_registry("{spec.modal_image}")
    .pip_install(
        "torch==2.5.1", "transformers>=4.44.0",
        "accelerate>=1.0.0", "safetensors>=0.4.0",
        "huggingface-hub[hf_transfer]>=0.25.0",
        "fastapi", "uvicorn",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

MINUTES = 60

@app.cls(
    image=image,
    gpu="{gpu_class}",
    timeout=5 * MINUTES,
    container_idle_timeout={idle},
    allow_concurrent_inputs=4,
    volumes={{
        "/root/.cache/huggingface": modal.Volume.from_name(
            "hf-cache-{spec.name}", create_if_missing=True
        ),
    }},
)
class Inference:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    @modal.web_endpoint(method="POST")
    def generate(self, request: dict):
        from fastapi.responses import JSONResponse
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 512)
        result = self.pipe(prompt, max_new_tokens=max_tokens)
        return JSONResponse({{"text": result[0]["generated_text"]}})

    @modal.web_endpoint(method="GET")
    def health(self):
        return {{"status": "ok", "model": MODEL_ID}}
'''


def generate_app_source(spec: ModelSpec, org: str) -> str:
    """Route to the right app generator based on engine + task."""
    if spec.inference_engine == "vllm" or spec.inference_engine == "sglang":
        return _vllm_app_source(spec, org)
    if spec.task_type == "image-generation":
        return _diffusers_image_app_source(spec, org)
    if spec.task_type == "video-generation":
        return _diffusers_video_app_source(spec, org)
    if spec.task_type == "tts":
        return _tts_app_source(spec, org)
    if spec.task_type == "stt":
        return _stt_app_source(spec, org)
    if spec.task_type == "music-generation":
        return _music_app_source(spec, org)
    return _generic_app_source(spec, org)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Modal model deployment orchestrator."""
    pass


@cli.command("list")
@click.option("--task", default=None, help="Filter by task type")
@click.option("--gpu", default=None, help="Filter by GPU type")
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def list_models(task: Optional[str], gpu: Optional[str], as_json: bool):
    """List all models in the registry."""
    models = filter_models(load_registry(), task=task, gpu=gpu)

    if as_json:
        import json
        print(json.dumps([m.__dict__ for m in models], indent=2))
        return

    # Table output
    header = f"{'Name':<28} {'Task':<20} {'GPU':<12} {'Count':<6} {'Engine':<14} {'VRAM MiB':<10}"
    print(header)
    print("-" * len(header))
    for m in sorted(models, key=lambda x: (x.task_type, x.name)):
        print(f"{m.name:<28} {m.task_type:<20} {m.gpu_type:<12} {m.gpu_count:<6} {m.inference_engine:<14} {m.vram_required_mib:<10}")
    print(f"\nTotal: {len(models)} models")


@cli.command("deploy")
@click.option("--model", default=None, help="Deploy specific model by name")
@click.option("--task", default=None, help="Deploy all models of a task type")
@click.option("--all", "deploy_all", is_flag=True, help="Deploy all models")
@click.option("--org", default="default", help="Modal organization name")
@click.option("--dry-run", is_flag=True, help="Generate app files without deploying")
@click.option("--out-dir", default=None, help="Output directory for generated apps (default: models/apps/)")
def deploy(
    model: Optional[str],
    task: Optional[str],
    deploy_all: bool,
    org: str,
    dry_run: bool,
    out_dir: Optional[str],
):
    """Deploy models to Modal."""
    if not any([model, task, deploy_all]):
        click.echo("Specify --model, --task, or --all", err=True)
        sys.exit(1)

    registry = load_registry()
    targets = filter_models(registry, name=model, task=task)
    if not targets:
        click.echo(f"No models matched filters (model={model}, task={task})", err=True)
        sys.exit(1)

    apps_dir = Path(out_dir) if out_dir else Path(__file__).parent / "apps"
    apps_dir.mkdir(parents=True, exist_ok=True)

    for spec in targets:
        source = generate_app_source(spec, org)
        app_file = apps_dir / f"{spec.name}.py"
        app_file.write_text(source)
        click.echo(f"Generated: {app_file}")

        if not dry_run:
            import subprocess
            click.echo(f"Deploying {spec.name}...")
            result = subprocess.run(
                ["modal", "deploy", str(app_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                click.echo(f"  OK: {spec.name}")
                # Extract URL from modal deploy output
                for line in result.stdout.splitlines():
                    if "modal.run" in line or "https://" in line:
                        click.echo(f"  URL: {line.strip()}")
            else:
                click.echo(f"  FAIL: {spec.name}", err=True)
                click.echo(result.stderr, err=True)


@cli.command("gen-config")
@click.option("--org", required=True, help="Modal organization name")
@click.option("--task", default=None, help="Filter by task type")
@click.option("--model", default=None, help="Filter by model name")
def gen_config(org: str, task: Optional[str], model: Optional[str]):
    """Generate operator.toml [[models]] entries for deployed models."""
    registry = load_registry()
    targets = filter_models(registry, name=model, task=task)

    for spec in targets:
        app_name = f"inference-{spec.name}"
        # Modal URL pattern: https://{org}--{app-name}-{cls}-{method}.modal.run
        # For cls-based apps, the URL includes the class and method name
        if spec.inference_engine in ("vllm", "sglang"):
            endpoint = f"https://{org}--{app_name}-serve.modal.run"
        else:
            endpoint = f"https://{org}--{app_name}-inference.modal.run"

        # Map task type to operator config type
        op_type = {
            "text-generation": "text",
            "image-generation": "image",
            "video-generation": "video",
            "tts": "tts",
            "stt": "stt",
            "music-generation": "music",
            "embedding": "embedding",
            "rerank": "rerank",
            "diarize": "diarize",
            "vad": "vad",
        }.get(spec.task_type, spec.task_type)

        pricing = ""
        if spec.price_per_m_tokens:
            pricing = f'price_per_1k = "{spec.price_per_m_tokens / 1000:.6f}"'
        elif spec.price_per_image:
            pricing = f'price_per_1k = "{spec.price_per_image}"'
        elif spec.price_per_second:
            pricing = f'price_per_1k = "{spec.price_per_second * 1000:.4f}"'

        print(f"""
[[models]]
name = "{spec.name}"
type = "{op_type}"
modal_endpoint = "{endpoint}"
{pricing}
health_path = "/health"
inference_path = "{spec.endpoint_path}"
""")


@cli.command("health")
@click.option("--org", required=True, help="Modal organization name")
@click.option("--task", default=None, help="Filter by task type")
def health_check(org: str, task: Optional[str]):
    """Health check deployed models."""
    import urllib.request
    import json

    registry = load_registry()
    targets = filter_models(registry, task=task)

    for spec in targets:
        app_name = f"inference-{spec.name}"
        if spec.inference_engine in ("vllm", "sglang"):
            url = f"https://{org}--{app_name}-serve.modal.run/health"
        else:
            url = f"https://{org}--{app_name}-inference-health.modal.run"

        try:
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            status = data.get("status", "unknown")
            click.echo(f"  {spec.name:<28} {status}")
        except Exception as e:
            click.echo(f"  {spec.name:<28} UNREACHABLE ({e})")


@cli.command("cost")
@click.option("--task", default=None, help="Filter by task type")
@click.option("--gpu", default=None, help="Filter by GPU type")
def estimate_cost(task: Optional[str], gpu: Optional[str]):
    """Estimate Modal GPU costs per model (idle container at min scale)."""
    # Modal GPU pricing (approximate $/hr as of 2025)
    gpu_hourly = {
        "T4": 0.59,
        "A10G": 1.10,
        "L4": 0.80,
        "L40S": 2.49,
        "A100-40GB": 3.22,
        "A100-80GB": 3.72,
        "H100": 4.76,
    }

    registry = load_registry()
    targets = filter_models(registry, task=task, gpu=gpu)

    header = f"{'Name':<28} {'GPU':<12} {'Count':<6} {'$/hr':<10} {'$/day':<10} {'$/mo (idle)':<12}"
    print(header)
    print("-" * len(header))

    total_hourly = 0.0
    for m in sorted(targets, key=lambda x: x.gpu_type):
        rate = gpu_hourly.get(m.gpu_type, 0) * m.gpu_count
        daily = rate * 24
        monthly = daily * 30
        total_hourly += rate
        print(f"{m.name:<28} {m.gpu_type:<12} {m.gpu_count:<6} ${rate:<9.2f} ${daily:<9.2f} ${monthly:<11.2f}")

    print(f"\n{'TOTAL':<28} {'':12} {'':6} ${total_hourly:<9.2f} ${total_hourly*24:<9.2f} ${total_hourly*24*30:<11.2f}")
    print("\nNote: Actual cost depends on idle timeout + scale-to-zero. These are max sustained costs.")


if __name__ == "__main__":
    cli()
