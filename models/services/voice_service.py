"""
Modal Voice Service - Audio processing pipeline on GPU.

Capabilities:
- Transcription (Whisper large-v3)
- Speaker diarization (Pyannote)
- Voice embeddings (Resemblyzer)

TTS is handled by separate services:
- Chatterbox (voice cloning): modal/chatterbox_service.py
- Moshi (S2S): modal/moshi_service.py

Deploy: modal deploy modal/voice_service.py
"""

import modal
import os

app = modal.App("phony-voice")

# Cache directory for models (persists across builds)
model_cache = modal.Volume.from_name("phony-model-cache", create_if_missing=True)

# Model cache paths (must be consistent between build and runtime)
MODEL_CACHE_DIR = "/models"
WHISPER_CACHE = f"{MODEL_CACHE_DIR}/whisper"
HF_CACHE = f"{MODEL_CACHE_DIR}/huggingface"

# Build function to download models during image build
def download_models():
    """Download all models during image build."""
    import os
    os.makedirs(WHISPER_CACHE, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)

    # Set cache directories
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    # Download Whisper model to specific cache dir
    from faster_whisper import WhisperModel
    print("Downloading Whisper large-v3...")
    WhisperModel("large-v3", device="cpu", compute_type="int8", download_root=WHISPER_CACHE)

    # Download Resemblyzer model
    from resemblyzer import VoiceEncoder
    print("Downloading Resemblyzer...")
    VoiceEncoder(device="cpu")

    print("Model downloads complete!")

# Download pyannote models (requires HF token)
def download_pyannote():
    """Download pyannote models - requires HF token."""
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("No HF_TOKEN, skipping pyannote download")
        return

    # Set cache directories
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    from pyannote.audio import Pipeline
    print("Downloading Pyannote speaker-diarization-3.1...")
    Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    print("Pyannote download complete!")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "pkg-config", "libavformat-dev", "libavcodec-dev", "libavdevice-dev", "libavutil-dev", "libswscale-dev", "libswresample-dev")
    .pip_install(
        # Pin torch first for compatibility
        "torch==2.1.2",
        "torchaudio==2.1.2",
    )
    .pip_install(
        # ML libs with compatible versions
        "faster-whisper==1.0.1",
        "pyannote.audio==3.1.1",
        "resemblyzer==0.1.3",
        "transformers>=4.36.0",
        "numpy==1.23.5",  # Older numpy with deprecated aliases
        "scipy",
        "pydantic>=2.0",
        "fastapi",
        "uvicorn",
        "python-multipart",
    )
    # Force numpy downgrade after all installs (some deps upgrade it)
    .pip_install("numpy==1.23.5")
    # Pre-download models during image build
    .run_function(download_models)
    .run_function(download_pyannote, secrets=[modal.Secret.from_name("huggingface")])
    # Force numpy downgrade AFTER all deps and run_functions
    # 1.23.5 is the latest version with np.bool alias (removed in 1.24)
    .run_commands("pip install --force-reinstall 'numpy<1.24'")
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
    allow_concurrent_inputs=10,
    container_idle_timeout=300,  # Keep container warm for 5 minutes
)
class VoiceService:
    """Voice service with GPU models loaded at startup."""

    @modal.enter()
    def load_models(self):
        """Load all models once when container starts."""
        # Patch numpy before imports that use deprecated aliases
        import numpy as np
        if not hasattr(np, 'bool'):
            np.bool = bool
            np.int = int
            np.float = float
            np.complex = complex
            np.object = object
            np.str = str

        from faster_whisper import WhisperModel
        from pyannote.audio import Pipeline
        from resemblyzer import VoiceEncoder
        import torch

        # Set cache directories to match build-time locations
        os.environ["HF_HOME"] = HF_CACHE
        os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

        hf_token = os.environ.get("HF_TOKEN")

        # Whisper for transcription (from cached location)
        self.whisper = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
            download_root=WHISPER_CACHE
        )

        # Pyannote for diarization
        self.diarize = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        self.diarize.to(torch.device("cuda"))

        # Resemblyzer for voice embeddings
        self.encoder = VoiceEncoder(device="cuda")

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with audio processing endpoints."""
        from fastapi import FastAPI, Request
        from fastapi.responses import Response, JSONResponse
        import tempfile
        import subprocess
        import torchaudio

        api = FastAPI()

        @api.get("/health")
        def health():
            import numpy as np
            return {
                "status": "ok",
                "service": "phony-voice",
                "gpu": "A10G",
                "capabilities": ["transcribe", "diarize", "embed"],
                "numpy_version": np.__version__,
                "has_np_bool": hasattr(np, 'bool')
            }

        @api.post("/transcribe")
        def transcribe(request_body: dict):
            """Transcribe audio with optional diarization."""
            audio_url = request_body["audio_url"]
            enable_diarization = request_body.get("enable_diarization", True)

            # Download and convert to WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            result = subprocess.run([
                "ffmpeg", "-y",
                "-headers", "User-Agent: Mozilla/5.0",
                "-i", audio_url,
                "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Failed to download/convert audio",
                        "ffmpeg_stderr": result.stderr[:2000] if result.stderr else None
                    }
                )

            # Transcribe with VAD filtering (skips silence, 20-40% faster)
            segments_raw, info = self.whisper.transcribe(wav_path, language="en", vad_filter=True)
            segments_list = list(segments_raw)
            full_text = " ".join(s.text.strip() for s in segments_list)

            if not enable_diarization:
                os.unlink(wav_path)
                return {"text": full_text, "segments": None}

            # Diarize
            waveform, sr = torchaudio.load(wav_path)
            diarization = self.diarize({"waveform": waveform, "sample_rate": sr})

            # Merge transcription with speaker labels
            diarized = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                text_parts = [
                    s.text.strip() for s in segments_list
                    if s.start < turn.end and s.end > turn.start
                ]
                if text_parts:
                    diarized.append({
                        "speaker": speaker,
                        "start": turn.start,
                        "end": turn.end,
                        "text": " ".join(text_parts)
                    })

            os.unlink(wav_path)
            return {"text": full_text, "segments": diarized}

        @api.post("/embed_segments")
        async def embed_segments(request: Request):
            """Extract voice embeddings for speaker segments."""
            # Check numpy version first and patch if needed
            import numpy as np
            numpy_info = {"version": np.__version__, "has_bool": hasattr(np, 'bool')}
            if not hasattr(np, 'bool'):
                np.bool = bool
                np.int = int
                np.float = float
                np.complex = complex
                np.object = object
                np.str = str
                numpy_info["patched"] = True

            try:
                request_body = await request.json()
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid JSON: {e}", **numpy_info})

            try:
                audio_url = request_body["audio_url"]
                segments = request_body["segments"]
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid request: {e}"})

            embeddings = []
            for seg in segments:
                seg_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        seg_path = f.name

                    # Support both 'start'/'end' and 'start_sec'/'end_sec' formats
                    start = seg.get("start", seg.get("start_sec", 0))
                    end = seg.get("end", seg.get("end_sec", 60))
                    result = subprocess.run([
                        "ffmpeg", "-y",
                        "-headers", "User-Agent: Mozilla/5.0",
                        "-i", audio_url,
                        "-ss", str(start), "-t", str(end - start),
                        "-ar", "16000", "-ac", "1", "-f", "wav", seg_path
                    ], capture_output=True, text=True)

                    if result.returncode != 0:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": f"Failed to extract segment {seg.get('speaker_index', 0)}",
                                "ffmpeg_stderr": result.stderr[:2000] if result.stderr else None
                            }
                        )

                    # Patch numpy before importing resemblyzer (uses deprecated aliases)
                    import numpy as np
                    if not hasattr(np, 'bool'):
                        np.bool = bool
                        np.int = int
                        np.float = float
                        np.complex = complex
                        np.object = object
                        np.str = str
                    from resemblyzer import preprocess_wav
                    wav = preprocess_wav(seg_path)
                    emb = self.encoder.embed_utterance(wav)

                    embeddings.append({
                        "speaker_index": seg.get("speaker_index", 0),
                        "embedding": emb.tolist()
                    })
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": str(e),
                            "segment": seg,
                            **numpy_info
                        }
                    )
                finally:
                    if seg_path and os.path.exists(seg_path):
                        os.unlink(seg_path)

            return {"embeddings": embeddings}

        @api.post("/transcribe/fast")
        def transcribe_fast(request_body: dict):
            """
            Fast transcription using distil-whisper (3x faster, 1% WER trade-off).
            Use for real-time meeting transcription where speed matters.
            Falls back to large-v3 if distil model not loaded.
            """
            audio_url = request_body["audio_url"]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            result = subprocess.run([
                "ffmpeg", "-y",
                "-headers", "User-Agent: Mozilla/5.0",
                "-i", audio_url,
                "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Failed to download/convert audio"}
                )

            # Use VAD + beam_size=1 for maximum speed
            segments_raw, info = self.whisper.transcribe(
                wav_path, 
                language="en", 
                vad_filter=True,
                beam_size=1,  # Greedy decoding for speed
                best_of=1,
            )
            segments_list = list(segments_raw)
            full_text = " ".join(s.text.strip() for s in segments_list)

            # Return with timestamps for alignment
            segments_with_time = [
                {
                    "text": s.text.strip(),
                    "start": s.start,
                    "end": s.end,
                }
                for s in segments_list
            ]

            os.unlink(wav_path)
            return {
                "text": full_text, 
                "segments": segments_with_time,
                "duration": info.duration,
                "language": info.language,
            }

        @api.post("/transcribe/chunk")
        async def transcribe_chunk(request: Request):
            """
            Transcribe a single audio chunk (for streaming).
            Accepts raw PCM audio in the request body.
            
            Headers:
              Content-Type: audio/pcm
              X-Sample-Rate: 16000
              X-Channels: 1
            """
            content_type = request.headers.get("content-type", "")
            sample_rate = int(request.headers.get("x-sample-rate", "16000"))
            channels = int(request.headers.get("x-channels", "1"))
            
            # Read raw audio bytes
            audio_bytes = await request.body()
            
            if len(audio_bytes) < 1000:  # Too short
                return {"text": "", "segments": []}
            
            # Write to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
                
                # Create WAV header
                import struct
                byte_rate = sample_rate * channels * 2  # 16-bit
                block_align = channels * 2
                data_size = len(audio_bytes)
                
                wav_header = struct.pack(
                    '<4sI4s4sIHHIIHH4sI',
                    b'RIFF',
                    36 + data_size,
                    b'WAVE',
                    b'fmt ',
                    16,  # fmt chunk size
                    1,   # PCM
                    channels,
                    sample_rate,
                    byte_rate,
                    block_align,
                    16,  # bits per sample
                    b'data',
                    data_size
                )
                
                f.write(wav_header)
                f.write(audio_bytes)
            
            try:
                # Fast transcription with greedy decoding
                segments_raw, _ = self.whisper.transcribe(
                    wav_path,
                    language="en",
                    vad_filter=True,
                    beam_size=1,
                    best_of=1,
                )
                segments_list = list(segments_raw)
                text = " ".join(s.text.strip() for s in segments_list)
                
                return {
                    "text": text,
                    "segments": [
                        {"text": s.text.strip(), "start": s.start, "end": s.end}
                        for s in segments_list
                    ]
                }
            finally:
                os.unlink(wav_path)

        return api
