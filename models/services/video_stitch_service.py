"""
Video Stitch Service — Concatenate multiple videos with transitions.

Runs on CPU (ffmpeg only, no GPU needed). Downloads input videos,
applies transitions, and returns a single concatenated MP4.

Endpoints:
  POST /stitch  — video_urls + transition → concatenated MP4
  GET  /health  — service status

Run:   modal run infra/modal-gpu/video_stitch_service.py
"""

import modal
import os

from base_service import create_modal_app, base_service_layer

app = create_modal_app("video-stitch")

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.11"))
    .apt_install("ffmpeg")
    .pip_install("fastapi", "uvicorn")
)


@app.cls(
    image=image,
    cpu=2,
    memory=4096,
    timeout=600,
    container_idle_timeout=120,
    allow_concurrent_inputs=10,
)
class VideoStitchService:
    """Concatenate videos via ffmpeg."""

    @modal.asgi_app()
    def web_app(self):
        import time
        import tempfile
        import subprocess
        import traceback
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response

        api = FastAPI()

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "ffmpeg-stitch",
                "gpu": "cpu",
                "features": ["stitch", "concat", "fade", "dissolve"],
            }

        @api.post("/stitch")
        async def stitch(request: Request):
            """Concatenate videos with optional transitions.

            Accepts JSON:
              - video_urls (list[str]): Ordered URLs of videos to concatenate
              - transition (str): 'cut', 'fade', or 'dissolve' (default 'cut')
              - transition_duration (float): Duration in seconds (default 0.5)

            Returns: MP4 video bytes
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}"})

            video_urls = body.get("video_urls", [])
            transition = body.get("transition", "cut")
            transition_duration = body.get("transition_duration", 0.5)

            if len(video_urls) < 2:
                return JSONResponse(status_code=400, content={"error": "At least 2 video URLs required"})
            if len(video_urls) > 50:
                return JSONResponse(status_code=400, content={"error": "Maximum 50 videos per stitch"})

            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "output.mp4")

            try:
                # Download all input videos
                input_paths = []
                for i, url in enumerate(video_urls):
                    path = os.path.join(temp_dir, f"input_{i}.mp4")
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-headers", "User-Agent: Mozilla/5.0",
                         "-i", url, "-c", "copy", path],
                        capture_output=True, timeout=120,
                    )
                    if result.returncode != 0:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Failed to download video {i}: {result.stderr.decode()[:200]}"},
                        )
                    input_paths.append(path)

                if transition == "cut":
                    _concat_cut(input_paths, output_path)
                elif transition in ("fade", "dissolve"):
                    _concat_with_transition(input_paths, output_path, transition, transition_duration)
                else:
                    return JSONResponse(status_code=400, content={"error": f"Unknown transition: {transition}"})

                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                elapsed = time.time() - start_time
                print(f"stitch: {len(video_urls)} videos → {len(video_bytes)} bytes in {elapsed:.1f}s")

                return Response(
                    content=video_bytes,
                    media_type="video/mp4",
                    headers={
                        "X-Generation-Time": f"{elapsed:.3f}",
                        "X-Input-Count": str(len(video_urls)),
                    },
                )
            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        return api


def _concat_cut(input_paths: list[str], output_path: str):
    """Simple concatenation — no transitions."""
    import tempfile

    # Create ffmpeg concat demuxer file
    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for path in input_paths:
            f.write(f"file '{path}'\n")

    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c", "copy", output_path],
        capture_output=True, timeout=300,
    )
    os.unlink(concat_file)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr.decode()[:500]}")


def _concat_with_transition(
    input_paths: list[str],
    output_path: str,
    transition: str,
    duration: float,
):
    """Concatenation with fade/dissolve transitions using xfade filter."""
    import subprocess

    # Build xfade filter chain
    # Each pair of adjacent videos gets an xfade filter
    n = len(input_paths)

    # First, get durations of each input
    durations = []
    for path in input_paths:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, timeout=30,
        )
        dur = float(result.stdout.decode().strip())
        durations.append(dur)

    # Build filter complex
    filter_parts = []
    offset = durations[0] - duration

    for i in range(1, n):
        prev = f"[v{i-1}]" if i > 1 else "[0:v]"
        curr = f"[{i}:v]"
        out = f"[v{i}]" if i < n - 1 else "[vout]"

        xfade_type = "fade" if transition == "fade" else "fadeblack"
        filter_parts.append(f"{prev}{curr}xfade=transition={xfade_type}:duration={duration}:offset={offset}{out}")

        if i < n - 1:
            offset += durations[i] - duration

    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"]
    for path in input_paths:
        cmd += ["-i", path]
    cmd += ["-filter_complex", filter_complex, "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", output_path]

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg xfade failed: {result.stderr.decode()[:500]}")


@app.local_entrypoint()
def main():
    print("Video Stitch Service ready")
    print("Deploy with: modal deploy infra/modal-gpu/video_stitch_service.py")
