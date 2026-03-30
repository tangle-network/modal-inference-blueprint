"""
Silero VAD v5 Service - Voice Activity Detection.

Capabilities:
- Voice activity detection with sub-millisecond latency per chunk
- Full-file and streaming speech segment detection
- 6000+ languages supported
- <1ms per 30ms chunk on CPU, 2MB model
- MIT license

Deploy: modal deploy modal/silero_vad_service.py

Requirements:
- CPU only (no GPU needed). Uses T4 for fast startup but model runs on CPU.
"""

import modal
import os

from base_service import create_modal_app, download_audio_to_tempfile

# --- Modal resources ---------------------------------------------------------

app = create_modal_app("silero-vad")

# --- Download model at image build time --------------------------------------

def download_silero_model():
    """Download Silero VAD v5 model during image build."""
    import torch

    print("Downloading Silero VAD v5...")
    torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        trust_repo=True,
    )
    print("Silero VAD v5 download complete.")


# --- Container image ---------------------------------------------------------

image = (
    base_service_layer(modal.Image.debian_slim(python_version="3.10"))
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "soundfile",
        "fastapi",
        "uvicorn",
        "silero-vad>=5.1",
    )
    .run_function(download_silero_model)
)


# --- Service class -----------------------------------------------------------

@app.cls(
    image=image,
    # CPU-bound workload; no GPU needed. Use cpu= for lower cost.
    cpu=2.0,
    memory=2048,
    timeout=300,
    container_idle_timeout=180,
    allow_concurrent_inputs=20,
)
class SileroVADService:
    """Voice activity detection via Silero VAD v5."""

    @modal.enter()
    def load_model(self):
        """Load Silero VAD model on container start."""
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

        self.model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        self._read_audio = read_audio

        print("Silero VAD v5 loaded (CPU).")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with VAD endpoints."""
        import time
        import traceback
        import base64
        import io
        import torch
        import torchaudio
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        api = FastAPI()
        svc = self

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "silero-vad-v5",
                "gpu": "none (cpu)",
                "capabilities": [
                    "voice_activity_detection",
                    "speech_timestamps",
                    "streaming_detection",
                    "6000_plus_languages",
                ],
                "supported_sample_rates": [8000, 16000],
                "chunk_size_ms": 32,
            }

        @api.post("/detect")
        async def detect(request: Request):
            """Detect speech segments in audio.

            Accepts JSON:
              - audio_url (str): URL to audio file
              - audio_file (str, base64): raw audio bytes (base64-encoded)
              - threshold (float, optional): speech probability threshold 0-1 (default 0.5)
              - min_speech_duration_ms (int, optional): min speech segment length (default 250)
              - min_silence_duration_ms (int, optional): min silence to split (default 100)
              - speech_pad_ms (int, optional): padding around speech segments (default 30)
              - return_seconds (bool, optional): return timestamps in seconds (default true)
              - sampling_rate (int, optional): override sample rate (default 16000)

            Returns:
              {
                "segments": [
                  {"start": 0.5, "end": 3.2, "duration": 2.7}
                ],
                "num_segments": 5,
                "total_speech_seconds": 45.3,
                "total_silence_seconds": 15.7,
                "speech_ratio": 0.74,
                "duration_seconds": 61.0,
                "processing_time": 0.05
              }
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"}
                )

            audio_url = body.get("audio_url")
            audio_b64 = body.get("audio_file")

            if not audio_url and not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_url or audio_file (base64)"},
                )

            threshold = body.get("threshold", 0.5)
            min_speech_ms = body.get("min_speech_duration_ms", 250)
            min_silence_ms = body.get("min_silence_duration_ms", 100)
            speech_pad_ms = body.get("speech_pad_ms", 30)
            return_seconds = body.get("return_seconds", True)
            sampling_rate = body.get("sampling_rate", 16000)

            temp_path = None
            try:
                audio_bytes = None
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)

                temp_path = download_audio_to_tempfile(
                    audio_url=audio_url,
                    audio_file=audio_bytes,
                    target_sr=sampling_rate,
                    mono=True,
                )

                wav = svc._read_audio(temp_path, sampling_rate=sampling_rate)

                timestamps = svc._get_speech_timestamps(
                    wav,
                    svc.model,
                    sampling_rate=sampling_rate,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_ms,
                    min_silence_duration_ms=min_silence_ms,
                    speech_pad_ms=speech_pad_ms,
                    return_seconds=return_seconds,
                )

                # Build structured segments
                segments = []
                total_speech = 0.0
                for ts in timestamps:
                    if return_seconds:
                        start_t = ts["start"]
                        end_t = ts["end"]
                    else:
                        start_t = ts["start"] / sampling_rate
                        end_t = ts["end"] / sampling_rate

                    dur = end_t - start_t
                    total_speech += dur
                    segments.append({
                        "start": round(start_t, 3),
                        "end": round(end_t, 3),
                        "duration": round(dur, 3),
                    })

                duration = len(wav) / sampling_rate
                total_silence = duration - total_speech
                speech_ratio = total_speech / duration if duration > 0 else 0

                elapsed = time.time() - start_time

                result = {
                    "segments": segments,
                    "num_segments": len(segments),
                    "total_speech_seconds": round(total_speech, 3),
                    "total_silence_seconds": round(total_silence, 3),
                    "speech_ratio": round(speech_ratio, 3),
                    "duration_seconds": round(duration, 3),
                    "processing_time": round(elapsed, 3),
                }

                rtf = elapsed / duration if duration > 0 else 0
                print(
                    f"silero-vad: detected {len(segments)} speech segments "
                    f"in {duration:.1f}s audio, {elapsed:.3f}s "
                    f"(RTF={rtf:.4f})"
                )

                return result

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        @api.post("/detect-stream")
        async def detect_stream(request: Request):
            """Process a streaming audio chunk for voice activity.

            Accepts JSON:
              - audio_chunk (str, base64): raw PCM audio chunk (base64-encoded)
              - sampling_rate (int, optional): sample rate (default 16000)
              - threshold (float, optional): speech probability threshold (default 0.5)
              - reset (bool, optional): reset VAD state before processing (default false)

            Returns:
              {
                "is_speech": true,
                "probability": 0.92,
                "processing_time": 0.001
              }

            Note: For streaming, send 512 samples (32ms) at 16kHz
            or 256 samples (32ms) at 8kHz per request.
            Call with reset=true at the start of a new audio stream.
            """
            start_time = time.time()

            try:
                body = await request.json()
            except Exception as e:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid JSON: {e}"}
                )

            audio_b64 = body.get("audio_chunk")
            if not audio_b64:
                return JSONResponse(
                    status_code=400,
                    content={"error": "audio_chunk (base64 PCM) is required"},
                )

            sampling_rate = body.get("sampling_rate", 16000)
            threshold = body.get("threshold", 0.5)
            reset = body.get("reset", False)

            try:
                import numpy as np

                if reset:
                    svc.model.reset_states()

                audio_bytes = base64.b64decode(audio_b64)

                # Interpret as 16-bit PCM
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(
                    np.float32
                ) / 32768.0
                chunk = torch.from_numpy(audio_np)

                # Get speech probability for this chunk
                speech_prob = svc.model(chunk, sampling_rate).item()

                elapsed = time.time() - start_time

                return {
                    "is_speech": speech_prob >= threshold,
                    "probability": round(speech_prob, 4),
                    "processing_time": round(elapsed, 6),
                }

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        return api


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Silero VAD v5 Service ready")
    print("Deploy with: modal deploy modal/silero_vad_service.py")
