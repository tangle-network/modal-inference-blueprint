"""
Modal deployment orchestrator — reads registry.toml, renders Jinja2 engine
templates, and deploys to Modal.

Usage:
    python models/deploy.py list                              # all models
    python models/deploy.py list --task video-avatar          # filter by task
    python models/deploy.py deploy --model hallo3 --org myorg # deploy one
    python models/deploy.py deploy --task tts --org myorg     # deploy all TTS
    python models/deploy.py deploy --all --org myorg --dry-run
    python models/deploy.py gen-config --org myorg            # operator.toml
    python models/deploy.py health --org myorg                # health check
    python models/deploy.py cost --task video-avatar          # GPU cost estimate
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from jinja2 import Environment, FileSystemLoader

REGISTRY_PATH = Path(__file__).parent / "registry.toml"
ENGINES_DIR = Path(__file__).parent / "engines"

# ---------------------------------------------------------------------------
# GPU + task config
# ---------------------------------------------------------------------------

GPU_MAP = {
    "T4": "T4", "A10G": "A10G", "L4": "L4", "L40S": "L40S",
    "A100-40GB": "A100", "A100-80GB": "A100-80GB", "H100": "H100",
}

IDLE_TIMEOUT = {
    "text-generation": 300, "image-generation": 180,
    "video-generation": 120, "video-avatar": 180, "video-lipsync": 180,
    "video-stitch": 120, "video-understanding": 300,
    "tts": 300, "stt": 300, "music-generation": 120,
    "audio-processing": 300, "voice-conversion": 180,
    "embedding": 600, "rerank": 600, "diarize": 300, "vad": 300,
}

CONCURRENCY = {
    "text-generation": 32, "image-generation": 4,
    "video-generation": 1, "video-avatar": 2, "video-lipsync": 2,
    "video-stitch": 10, "video-understanding": 8,
    "tts": 8, "stt": 16, "music-generation": 2,
    "audio-processing": 16, "voice-conversion": 4,
    "embedding": 64, "rerank": 64, "diarize": 4, "vad": 16,
}

# Engine template selection: task_type → template file (without .py.j2)
# inference_engine overrides when "vllm" or "sglang"
ENGINE_MAP = {
    "text-generation": "generic",
    "image-generation": "diffusers_image",
    "video-generation": "diffusers_video",
    "video-avatar": "video_avatar",
    "video-lipsync": "video_lipsync",
    "video-stitch": "video_stitch",
    "video-understanding": "video_understanding",
    "tts": "tts",
    "stt": "stt",
    "music-generation": "music",
    "audio-processing": "generic",
    "voice-conversion": "generic",
    "embedding": "generic",
    "rerank": "generic",
    "diarize": "generic",
    "vad": "generic",
}

# Diffusers pipeline class resolution
DIFFUSERS_VIDEO_PIPELINES = {
    "wan": "WanPipeline", "cogvideo": "CogVideoXPipeline",
    "ltx": "LTXPipeline", "hunyuan": "HunyuanVideoPipeline",
}

DIFFUSERS_IMAGE_PIPELINES = {
    "flux": "FluxPipeline", "sd3": "StableDiffusion3Pipeline",
    "stable-diffusion-3": "StableDiffusion3Pipeline",
    "sdxl": "StableDiffusionXLPipeline", "pixart": "PixArtSigmaPipeline",
    "kolors": "KolorsPipeline",
}

GPU_HOURLY = {
    "T4": 0.59, "A10G": 1.10, "L4": 0.80, "L40S": 2.49,
    "A100-40GB": 3.22, "A100-80GB": 3.72, "H100": 4.76,
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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


def filter_models(models: list[ModelSpec], name: str | None = None,
                  task: str | None = None, gpu: str | None = None) -> list[ModelSpec]:
    result = models
    if name:
        result = [m for m in result if m.name == name]
    if task:
        result = [m for m in result if m.task_type == task]
    if gpu:
        result = [m for m in result if m.gpu_type == gpu]
    return result


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _resolve_pipeline_cls(spec: ModelSpec, pipelines: dict[str, str], default: str) -> str:
    model_lower = spec.model_id.lower() + spec.name.lower()
    for key, cls in pipelines.items():
        if key in model_lower:
            return cls
    return default


def render_app(spec: ModelSpec, org: str) -> str:
    """Render a Modal app from a Jinja2 engine template."""
    # Pick engine template
    if spec.inference_engine in ("vllm", "sglang"):
        engine = "vllm"
    else:
        engine = ENGINE_MAP.get(spec.task_type, "generic")

    env = Environment(loader=FileSystemLoader(str(ENGINES_DIR)), keep_trailing_newline=True)
    template = env.get_template(f"{engine}.py.j2")

    gpu_class = GPU_MAP.get(spec.gpu_type, spec.gpu_type)
    gpu_spec = (f'"{gpu_class}"' if spec.gpu_count <= 1
                else f'modal.gpu.{gpu_class}(count={spec.gpu_count})')
    idle = IDLE_TIMEOUT.get(spec.task_type, 300)
    concurrency = CONCURRENCY.get(spec.task_type, 4)

    # Diffusers pipeline class
    pipeline_cls = "DiffusionPipeline"
    if spec.task_type == "video-generation":
        pipeline_cls = _resolve_pipeline_cls(spec, DIFFUSERS_VIDEO_PIPELINES, "DiffusionPipeline")
    elif spec.task_type == "image-generation":
        pipeline_cls = _resolve_pipeline_cls(spec, DIFFUSERS_IMAGE_PIPELINES, "AutoPipelineForText2Image")

    return template.render(
        spec=spec, org=org, gpu_class=gpu_class, gpu_spec=gpu_spec,
        idle=idle, concurrency=concurrency, pipeline_cls=pipeline_cls,
    )


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
def list_models(task: str | None, gpu: str | None, as_json: bool):
    """List all models in the registry."""
    models = filter_models(load_registry(), task=task, gpu=gpu)
    if as_json:
        import json
        print(json.dumps([m.__dict__ for m in models], indent=2))
        return
    header = f"{'Name':<28} {'Task':<22} {'GPU':<12} {'#':<4} {'Engine':<14} {'VRAM':<8}"
    print(header)
    print("-" * len(header))
    for m in sorted(models, key=lambda x: (x.task_type, x.name)):
        print(f"{m.name:<28} {m.task_type:<22} {m.gpu_type:<12} {m.gpu_count:<4} {m.inference_engine:<14} {m.vram_required_mib:<8}")
    print(f"\nTotal: {len(models)} models")


@cli.command("deploy")
@click.option("--model", default=None, help="Deploy specific model")
@click.option("--task", default=None, help="Deploy all models of a task type")
@click.option("--all", "deploy_all", is_flag=True, help="Deploy all models")
@click.option("--org", default="default", help="Modal organization name")
@click.option("--dry-run", is_flag=True, help="Generate without deploying")
@click.option("--out-dir", default=None, help="Output directory (default: models/apps/)")
def deploy(model: str | None, task: str | None, deploy_all: bool,
           org: str, dry_run: bool, out_dir: str | None):
    """Deploy models to Modal."""
    if not any([model, task, deploy_all]):
        click.echo("Specify --model, --task, or --all", err=True)
        sys.exit(1)

    targets = filter_models(load_registry(), name=model, task=task)
    if not targets:
        click.echo(f"No models matched (model={model}, task={task})", err=True)
        sys.exit(1)

    apps_dir = Path(out_dir) if out_dir else Path(__file__).parent / "apps"
    apps_dir.mkdir(parents=True, exist_ok=True)

    for spec in targets:
        source = render_app(spec, org)
        app_file = apps_dir / f"{spec.name}.py"
        app_file.write_text(source)
        click.echo(f"Generated: {app_file}")

        if not dry_run:
            import subprocess
            click.echo(f"Deploying {spec.name}...")
            result = subprocess.run(["modal", "deploy", str(app_file)], capture_output=True, text=True)
            if result.returncode == 0:
                click.echo(f"  OK: {spec.name}")
                for line in result.stdout.splitlines():
                    if "https://" in line:
                        click.echo(f"  URL: {line.strip()}")
            else:
                click.echo(f"  FAIL: {spec.name}", err=True)
                click.echo(result.stderr, err=True)


@cli.command("gen-config")
@click.option("--org", required=True, help="Modal organization name")
@click.option("--task", default=None, help="Filter by task type")
@click.option("--model", default=None, help="Filter by model name")
def gen_config(org: str, task: str | None, model: str | None):
    """Generate operator.toml [[models]] entries."""
    targets = filter_models(load_registry(), name=model, task=task)
    for spec in targets:
        app_name = f"inference-{spec.name}"
        if spec.inference_engine in ("vllm", "sglang"):
            endpoint = f"https://{org}--{app_name}-serve.modal.run"
        else:
            endpoint = f"https://{org}--{app_name}-inference.modal.run"

        pricing = ""
        if spec.price_per_m_tokens:
            pricing = f'price_per_1k = "{spec.price_per_m_tokens / 1000:.6f}"'
        elif spec.price_per_image:
            pricing = f'price_per_1k = "{spec.price_per_image}"'
        elif spec.price_per_second:
            pricing = f'price_per_1k = "{spec.price_per_second * 1000:.4f}"'

        print(f'\n[[models]]\nname = "{spec.name}"\ntype = "{spec.task_type}"')
        print(f'modal_endpoint = "{endpoint}"')
        if pricing:
            print(pricing)
        print(f'health_path = "/health"\ninference_path = "{spec.endpoint_path}"')


@cli.command("health")
@click.option("--org", required=True, help="Modal organization name")
@click.option("--task", default=None, help="Filter by task type")
def health_check(org: str, task: str | None):
    """Health check deployed models."""
    import urllib.request, json
    targets = filter_models(load_registry(), task=task)
    for spec in targets:
        app_name = f"inference-{spec.name}"
        if spec.inference_engine in ("vllm", "sglang"):
            url = f"https://{org}--{app_name}-serve.modal.run/health"
        else:
            url = f"https://{org}--{app_name}-inference-health.modal.run"
        try:
            resp = urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=10)
            data = json.loads(resp.read())
            click.echo(f"  {spec.name:<28} {data.get('status', '?')}")
        except Exception as e:
            click.echo(f"  {spec.name:<28} UNREACHABLE ({e})")


@cli.command("cost")
@click.option("--task", default=None, help="Filter by task type")
@click.option("--gpu", default=None, help="Filter by GPU type")
def estimate_cost(task: str | None, gpu: str | None):
    """Estimate Modal GPU costs per model."""
    targets = filter_models(load_registry(), task=task, gpu=gpu)
    header = f"{'Name':<28} {'GPU':<12} {'#':<4} {'$/hr':<10} {'$/day':<10} {'$/mo':<12}"
    print(header)
    print("-" * len(header))
    total = 0.0
    for m in sorted(targets, key=lambda x: x.gpu_type):
        rate = GPU_HOURLY.get(m.gpu_type, 0) * max(m.gpu_count, 1)
        total += rate
        print(f"{m.name:<28} {m.gpu_type:<12} {m.gpu_count:<4} ${rate:<9.2f} ${rate*24:<9.2f} ${rate*24*30:<11.2f}")
    print(f"\n{'TOTAL':<28} {'':12} {'':4} ${total:<9.2f} ${total*24:<9.2f} ${total*24*30:<11.2f}")
    print("\nNote: Actual cost depends on idle timeout + scale-to-zero.")


if __name__ == "__main__":
    cli()
