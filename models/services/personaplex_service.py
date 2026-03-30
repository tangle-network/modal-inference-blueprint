"""
PersonaPlex S2S Service - NVIDIA's full-duplex voice AI with persona control.

Based on Moshi architecture (Kyutai), PersonaPlex enables:
- Real-time speech-to-speech conversations
- Persona control via text prompts
- Voice selection from predefined embeddings
- Custom voice cloning from audio samples
- Full-duplex (can listen while speaking)

Model: nvidia/personaplex-7b-v1 (7B parameters, ~14GB VRAM)
Audio: 24kHz sample rate, Opus codec for streaming

Voice IDs available:
- NATF0, NATF1, NATF2, NATF3 (natural female)
- NATM0, NATM1, NATM2, NATM3 (natural male)
- VARF0, VARF1, VARF2, VARF3, VARF4 (variety female)
- VARM0, VARM1, VARM2, VARM3, VARM4 (variety male)
- Custom voices via /voice/clone endpoint

Voice Cloning Flow:
1. POST /voice/clone with audio sample -> returns voice_id
2. Connect to /ws with voice_id or voice_embedding
3. Speak naturally with your cloned voice

Deploy: modal deploy modal/personaplex_service.py
"""

import modal
import os
import uuid
import base64
from typing import Optional

app = modal.App("phony-personaplex")

# Volume for storing custom voice embeddings (model weights are baked into image)
voice_embeddings_cache = modal.Volume.from_name("phony-personaplex-voices", create_if_missing=True)

# Model weights baked into image at build time (don't mount a volume here!)
PERSONAPLEX_DIR = "/models/personaplex"
MOSHI_DIR = "/opt/personaplex"
VOICE_EMBEDDINGS_DIR = "/voice-embeddings"
SAMPLE_RATE = 24000
FRAME_RATE = 12.5  # Mimi codec frame rate (24000 / 1920)

# Model file names (from loaders.py constants)
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
LM_FILE = "model.safetensors"
TOKENIZER_FILE = "tokenizer_spm_32k_3.model"

# Audio constraints for voice cloning
MIN_AUDIO_DURATION = 3.0  # Minimum 3 seconds of audio
MAX_AUDIO_DURATION = 30.0  # Maximum 30 seconds of audio
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "ogg", "flac", "m4a", "webm"]

# Available voice embeddings
VOICE_IDS = [
    "NATF0", "NATF1", "NATF2", "NATF3",  # Natural female
    "NATM0", "NATM1", "NATM2", "NATM3",  # Natural male
    "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",  # Variety female
    "VARM0", "VARM1", "VARM2", "VARM3", "VARM4",  # Variety male
]


def download_models():
    """Download PersonaPlex model during image build."""
    import os
    from huggingface_hub import snapshot_download

    # Create parent directory
    os.makedirs("/models", exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: No HF_TOKEN, model download may fail for gated repo")

    print("Downloading nvidia/personaplex-7b-v1...")
    snapshot_download(
        repo_id="nvidia/personaplex-7b-v1",
        local_dir=PERSONAPLEX_DIR,
        token=hf_token,
    )
    print("PersonaPlex model downloaded!")


def wrap_with_system_tags(text: str) -> str:
    """Wrap text prompt with system tags for PersonaPlex."""
    return f"<system> {text} <system>"


# Build timestamp: 2026-01-26T17:00 - force rebuild
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "libopus-dev",  # Required for Opus codec
        "pkg-config",
        "libsndfile1",
    )
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "numpy<2",
        "scipy",
        "fastapi",
        "uvicorn",
        "websockets",
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
        "aiohttp",
        "sphn",  # Opus streaming codec
        "soundfile",
        "python-multipart",  # Required for FastAPI file uploads
        "pyloudnorm",  # Required for audio normalization in voice cloning
    )
    # Clone PersonaPlex repo and install moshi
    .run_commands(
        "git clone https://github.com/NVIDIA/personaplex.git /opt/personaplex",
        "cd /opt/personaplex && pip install moshi/.",
    )
    # Download model weights
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface")],
    )
)


@app.cls(
    image=image,
    gpu="A10G",  # 24GB VRAM, sufficient for 7B model
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=60,  # Scale down after 1 min idle (cost savings for dev)
    # min_containers=1,  # EXPENSIVE! Only enable for production
    volumes={
        VOICE_EMBEDDINGS_DIR: voice_embeddings_cache,  # Only mount voice embeddings volume
    },
)
class PersonaPlexService:
    """PersonaPlex S2S service with WebSocket streaming."""

    @modal.enter()
    def load_model(self):
        """Load PersonaPlex model on container startup."""
        import sys
        import time
        import torch
        import sentencepiece

        start_time = time.time()

        # Add moshi to path
        sys.path.insert(0, f"{MOSHI_DIR}/moshi")

        from moshi.models import loaders

        print(f"[{time.time()-start_time:.1f}s] Loading PersonaPlex model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{time.time()-start_time:.1f}s] Device: {self.device}")

        # Load Mimi audio codec - two instances as per original server:
        # - mimi: for decoding model output to audio
        # - other_mimi: for encoding user input to codes
        # Construct full file paths
        mimi_path = os.path.join(PERSONAPLEX_DIR, MIMI_FILE)
        lm_path = os.path.join(PERSONAPLEX_DIR, LM_FILE)
        tokenizer_path = os.path.join(PERSONAPLEX_DIR, TOKENIZER_FILE)

        print(f"[{time.time()-start_time:.1f}s] Loading Mimi codec #1 from {mimi_path}...")
        self.mimi = loaders.get_mimi(mimi_path, device=self.device)
        self.mimi.set_num_codebooks(8)
        print(f"[{time.time()-start_time:.1f}s] Mimi #1 loaded")

        print(f"[{time.time()-start_time:.1f}s] Loading Mimi codec #2...")
        self.other_mimi = loaders.get_mimi(mimi_path, device=self.device)
        self.other_mimi.set_num_codebooks(8)
        print(f"[{time.time()-start_time:.1f}s] Mimi #2 loaded")

        # Load LM model
        print(f"[{time.time()-start_time:.1f}s] Loading PersonaPlex LM (7B) from {lm_path}...")
        self.lm = loaders.get_moshi_lm(lm_path, device=self.device)
        print(f"[{time.time()-start_time:.1f}s] LM loaded")

        # Load text tokenizer (sentencepiece)
        print(f"[{time.time()-start_time:.1f}s] Loading tokenizer from {tokenizer_path}...")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        print(f"[{time.time()-start_time:.1f}s] Tokenizer loaded")

        # Load voice embeddings directory (preset voices)
        self.voice_prompt_dir = f"{MOSHI_DIR}/assets/voice_prompts"

        # Custom voice embeddings directory
        self.custom_voice_dir = f"{VOICE_EMBEDDINGS_DIR}/custom"
        os.makedirs(self.custom_voice_dir, exist_ok=True)

        # In-memory cache for loaded embeddings (avoid disk reads)
        self._embedding_cache: dict = {}
        self._max_cache_size = 10

        print(f"[{time.time()-start_time:.1f}s] PersonaPlex loaded on {self.device}")
        print(f"Mimi sample_rate: {self.mimi.sample_rate}, frame_rate: {self.mimi.frame_rate}")

        # Warmup the model with dummy inference (as per original server)
        print(f"[{time.time()-start_time:.1f}s] Starting warmup...")
        self._warmup()
        print(f"[{time.time()-start_time:.1f}s] Model ready! Total startup: {time.time()-start_time:.1f}s")

    def _warmup(self):
        """Warm up the model with dummy inference to stabilize GPU."""
        import torch
        from moshi.models import LMGen

        # Create a temporary LMGen for warmup
        lm_gen = LMGen(
            self.lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=self.device,
            frame_rate=self.mimi.frame_rate,
        )

        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        # Run 4 warmup iterations with dummy audio
        dummy_audio = torch.zeros(1, 1, 1920, device=self.device)
        for _ in range(4):
            with torch.no_grad():
                codes = self.other_mimi.encode(dummy_audio)
                for t in range(codes.shape[-1]):
                    _ = lm_gen.step(codes[:, :, t:t+1])

        # Reset streaming state after warmup
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()

    def _get_voice_prompt_path(self, voice_id: str) -> tuple[str, bool]:
        """Get path to voice embedding file.

        Returns:
            Tuple of (path, is_embedding) where is_embedding indicates
            if it's a pre-computed .pt embedding file.
        """
        # Check preset voices first
        if voice_id in VOICE_IDS:
            voice_path = os.path.join(self.voice_prompt_dir, f"{voice_id}.pt")
            if os.path.exists(voice_path):
                return voice_path, True

        # Check custom voices (always .pt embeddings)
        custom_path = os.path.join(self.custom_voice_dir, f"{voice_id}.pt")
        if os.path.exists(custom_path):
            return custom_path, True

        # Not found
        available = VOICE_IDS + self._list_custom_voices()
        raise ValueError(f"Voice ID '{voice_id}' not found. Available: {available}")

    def _list_custom_voices(self) -> list[str]:
        """List all custom voice IDs."""
        custom_voices = []
        if os.path.exists(self.custom_voice_dir):
            for f in os.listdir(self.custom_voice_dir):
                if f.endswith(".pt"):
                    custom_voices.append(f[:-3])  # Remove .pt extension
        return custom_voices

    def _encode_voice_embedding(self, audio_path: str) -> dict:
        """Encode audio file into voice embedding using Mimi.

        This creates a voice embedding that can be used with PersonaPlex.
        The embedding is computed by running the audio through the Mimi
        encoder and capturing the model's intermediate representations.

        Args:
            audio_path: Path to audio file (WAV format, 24kHz recommended)

        Returns:
            Dictionary with 'embeddings' and 'cache' tensors ready for torch.save
        """
        import torch
        import numpy as np
        import subprocess
        import tempfile

        from moshi.models import LMGen

        # Convert audio to proper format if needed (24kHz mono WAV)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            converted_path = f.name

        try:
            # Use ffmpeg to ensure proper format
            result = subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-f", "wav", converted_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise ValueError(f"Audio conversion failed: {result.stderr[:500]}")

            # Validate duration
            import soundfile as sf
            info = sf.info(converted_path)
            duration = info.duration

            if duration < MIN_AUDIO_DURATION:
                raise ValueError(f"Audio too short ({duration:.1f}s). Minimum is {MIN_AUDIO_DURATION}s.")

            if duration > MAX_AUDIO_DURATION:
                # Truncate to max duration
                print(f"Audio truncated from {duration:.1f}s to {MAX_AUDIO_DURATION}s")
                truncated_path = converted_path + "_truncated.wav"
                subprocess.run([
                    "ffmpeg", "-y", "-i", converted_path,
                    "-t", str(MAX_AUDIO_DURATION),
                    "-f", "wav", truncated_path
                ], capture_output=True)
                os.unlink(converted_path)
                converted_path = truncated_path

            # Create LMGen with save_voice_prompt_embeddings enabled
            # PersonaPlex saves embeddings to a .pt file at the same path as the audio
            lm_gen = LMGen(
                self.lm,
                audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                sample_rate=self.mimi.sample_rate,
                device=self.device,
                frame_rate=self.mimi.frame_rate,
                save_voice_prompt_embeddings=True,  # This saves to {audio_path}.pt
            )

            # Set up streaming
            lm_gen.streaming_forever(1)
            self.other_mimi.reset_streaming()
            self.other_mimi.streaming_forever(1)

            # Load voice prompt (this will encode the audio)
            lm_gen.load_voice_prompt(converted_path)

            # Process voice prompt to generate embeddings
            # This runs through the model and saves embeddings to a .pt file
            lm_gen._step_voice_prompt(self.other_mimi)

            # Read the saved embeddings file (PersonaPlex saves to {audio_path_without_ext}.pt)
            from os.path import splitext
            embedding_path = splitext(converted_path)[0] + ".pt"

            if not os.path.exists(embedding_path):
                raise ValueError(f"Embedding file not created at {embedding_path}")

            saved_data = torch.load(embedding_path, map_location="cpu")
            embeddings = saved_data["embeddings"]
            cache = saved_data["cache"]

            # Clean up the temp embedding file
            os.unlink(embedding_path)

            # Reset streaming state
            self.other_mimi.reset_streaming()

            return {
                "embeddings": embeddings,
                "cache": cache,
            }

        finally:
            # Clean up temp files
            if os.path.exists(converted_path):
                os.unlink(converted_path)
            truncated_path = converted_path + "_truncated.wav"
            if os.path.exists(truncated_path):
                os.unlink(truncated_path)

    def _save_voice_embedding(self, voice_id: str, embedding_data: dict) -> str:
        """Save voice embedding to volume."""
        import torch

        voice_path = os.path.join(self.custom_voice_dir, f"{voice_id}.pt")
        torch.save(embedding_data, voice_path)

        # Commit volume changes
        voice_embeddings_cache.commit()

        return voice_path

    def _load_voice_embedding_from_base64(self, embedding_b64: str) -> dict:
        """Load voice embedding from base64-encoded string."""
        import torch
        import io

        embedding_bytes = base64.b64decode(embedding_b64)
        buffer = io.BytesIO(embedding_bytes)
        return torch.load(buffer, map_location=self.device)

    @modal.asgi_app()
    def web_app(self):
        """FastAPI app with WebSocket endpoint for S2S streaming and voice cloning."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
        from fastapi.responses import JSONResponse, Response
        import torch
        import numpy as np
        import io
        import tempfile
        import time

        api = FastAPI()

        @api.get("/health")
        def health():
            custom_voices = self._list_custom_voices()
            return {
                "status": "ok",
                "service": "personaplex-s2s",
                "model": "nvidia/personaplex-7b-v1",
                "sample_rate": SAMPLE_RATE,
                "voices": VOICE_IDS,
                "custom_voices": custom_voices,
                "features": ["voice_cloning", "voice_embedding", "full_duplex"],
            }

        @api.get("/voices")
        def list_voices():
            """List available voice embeddings (preset and custom)."""
            preset_voices = [
                {"id": v, "gender": "female" if "F" in v else "male",
                 "type": "natural" if v.startswith("NAT") else "variety"}
                for v in VOICE_IDS
            ]
            custom_voices = [
                {"id": v, "type": "custom"}
                for v in self._list_custom_voices()
            ]
            return {
                "preset_voices": preset_voices,
                "custom_voices": custom_voices,
                "voices": preset_voices + custom_voices,  # Combined for compatibility
            }

        @api.post("/voice/encode")
        async def encode_voice(
            audio_file: UploadFile = File(None),
            audio_base64: str = Form(None),
            audio_url: str = Form(None),
        ):
            """Encode audio into a voice embedding.

            Accepts audio via:
            - audio_file: File upload (WAV, MP3, etc.)
            - audio_base64: Base64-encoded audio data
            - audio_url: URL to download audio from

            Returns:
            - embedding_base64: Base64-encoded .pt file that can be used with /ws
            - duration: Duration of processed audio
            - message: Status message
            """
            start_time = time.time()

            if not audio_file and not audio_base64 and not audio_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_file, audio_base64, or audio_url"}
                )

            try:
                # Get audio data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                if audio_file:
                    audio_data = await audio_file.read()
                    with open(temp_path, "wb") as f:
                        f.write(audio_data)
                elif audio_base64:
                    audio_data = base64.b64decode(audio_base64)
                    with open(temp_path, "wb") as f:
                        f.write(audio_data)
                elif audio_url:
                    import subprocess
                    result = subprocess.run([
                        "ffmpeg", "-y",
                        "-headers", "User-Agent: Mozilla/5.0",
                        "-i", audio_url,
                        "-ar", str(SAMPLE_RATE), "-ac", "1",
                        "-t", str(MAX_AUDIO_DURATION),
                        "-f", "wav", temp_path
                    ], capture_output=True, text=True)
                    if result.returncode != 0:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Failed to download audio: {result.stderr[:500]}"}
                        )

                # Encode the voice embedding
                embedding_data = self._encode_voice_embedding(temp_path)

                # Serialize to bytes
                buffer = io.BytesIO()
                torch.save(embedding_data, buffer)
                buffer.seek(0)
                embedding_bytes = buffer.read()
                embedding_base64 = base64.b64encode(embedding_bytes).decode("utf-8")

                elapsed = time.time() - start_time
                print(f"Voice encoding completed in {elapsed:.2f}s")

                return {
                    "status": "success",
                    "embedding_base64": embedding_base64,
                    "embedding_size_bytes": len(embedding_bytes),
                    "encoding_time": elapsed,
                    "message": "Voice embedding created. Use with /ws via voice_embedding parameter."
                }

            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        @api.post("/voice/clone")
        async def clone_voice(
            voice_id: str = Form(None),
            audio_file: UploadFile = File(None),
            audio_base64: str = Form(None),
            audio_url: str = Form(None),
        ):
            """Create and save a custom cloned voice.

            Accepts audio via:
            - audio_file: File upload (WAV, MP3, etc.)
            - audio_base64: Base64-encoded audio data
            - audio_url: URL to download audio from

            Parameters:
            - voice_id: Optional custom ID. If not provided, generates a UUID.

            Returns:
            - voice_id: The ID to use with /ws
            - message: Status message

            Usage:
            1. Call this endpoint with audio sample (10-30 seconds recommended)
            2. Use returned voice_id in WebSocket connection: {"voice_id": "your_voice_id"}
            """
            start_time = time.time()

            if not audio_file and not audio_base64 and not audio_url:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Provide audio_file, audio_base64, or audio_url"}
                )

            # Generate voice_id if not provided
            if not voice_id:
                voice_id = f"custom_{uuid.uuid4().hex[:12]}"

            # Validate voice_id format
            if voice_id in VOICE_IDS:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Cannot use preset voice ID '{voice_id}'. Choose a different name."}
                )

            try:
                # Get audio data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                if audio_file:
                    audio_data = await audio_file.read()
                    with open(temp_path, "wb") as f:
                        f.write(audio_data)
                elif audio_base64:
                    audio_data = base64.b64decode(audio_base64)
                    with open(temp_path, "wb") as f:
                        f.write(audio_data)
                elif audio_url:
                    import subprocess
                    result = subprocess.run([
                        "ffmpeg", "-y",
                        "-headers", "User-Agent: Mozilla/5.0",
                        "-i", audio_url,
                        "-ar", str(SAMPLE_RATE), "-ac", "1",
                        "-t", str(MAX_AUDIO_DURATION),
                        "-f", "wav", temp_path
                    ], capture_output=True, text=True)
                    if result.returncode != 0:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Failed to download audio: {result.stderr[:500]}"}
                        )

                # Encode the voice embedding
                embedding_data = self._encode_voice_embedding(temp_path)

                # Save to volume
                voice_path = self._save_voice_embedding(voice_id, embedding_data)

                elapsed = time.time() - start_time
                print(f"Voice cloning completed in {elapsed:.2f}s for voice_id={voice_id}")

                return {
                    "status": "success",
                    "voice_id": voice_id,
                    "cloning_time": elapsed,
                    "message": f"Voice '{voice_id}' cloned successfully. Use in WebSocket with voice_id parameter."
                }

            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        @api.delete("/voice/{voice_id}")
        async def delete_voice(voice_id: str):
            """Delete a custom cloned voice."""
            if voice_id in VOICE_IDS:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Cannot delete preset voice '{voice_id}'"}
                )

            voice_path = os.path.join(self.custom_voice_dir, f"{voice_id}.pt")
            if not os.path.exists(voice_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Voice '{voice_id}' not found"}
                )

            try:
                os.unlink(voice_path)
                # Remove from cache if present
                if voice_id in self._embedding_cache:
                    del self._embedding_cache[voice_id]
                voice_embeddings_cache.commit()
                return {"status": "success", "message": f"Voice '{voice_id}' deleted"}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @api.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            """WebSocket endpoint for real-time S2S conversation.

            Protocol:
            1. Client connects
            2. Client sends init JSON with one of:
               - {"persona": "...", "voice_id": "NATM1"} - Use preset or cloned voice by ID
               - {"persona": "...", "voice_embedding": "base64..."} - Use inline embedding
            3. Client sends binary audio: 24kHz PCM int16
            4. Server responds with binary audio: 24kHz PCM int16

            The server processes audio in streaming chunks, enabling
            full-duplex conversation with ~200ms latency.

            Voice options:
            - voice_id: Preset (NATM1, etc.) or custom cloned voice ID
            - voice_embedding: Base64-encoded .pt embedding from /voice/encode
            """
            await ws.accept()
            print("WebSocket connection accepted")

            initialized = False
            lm_gen = None

            # Audio buffers - Mimi uses 1920 samples per frame (80ms at 24kHz)
            input_buffer = bytearray()
            FRAME_SAMPLES = 1920  # 80ms at 24kHz
            BYTES_PER_FRAME = FRAME_SAMPLES * 2  # int16 = 2 bytes per sample

            try:
                while True:
                    message = await ws.receive()

                    if "text" in message:
                        # JSON configuration message
                        import json
                        try:
                            config = json.loads(message["text"])
                            persona = config.get("persona", "You are a helpful assistant.")
                            voice_id = config.get("voice_id")
                            voice_embedding_b64 = config.get("voice_embedding")

                            # Determine voice source
                            using_embedding = False
                            voice_path = None
                            embedding_data = None

                            if voice_embedding_b64:
                                # Use inline base64 embedding
                                print(f"Initializing with persona='{persona[:50]}...', inline embedding")
                                try:
                                    embedding_data = self._load_voice_embedding_from_base64(voice_embedding_b64)
                                    using_embedding = True
                                except Exception as e:
                                    await ws.send_json({"error": f"Invalid voice_embedding: {e}"})
                                    continue
                            elif voice_id:
                                # Use voice_id (preset or custom cloned)
                                print(f"Initializing with persona='{persona[:50]}...', voice_id={voice_id}")
                                try:
                                    voice_path, is_embedding = self._get_voice_prompt_path(voice_id)
                                    using_embedding = is_embedding
                                except ValueError as e:
                                    await ws.send_json({"error": str(e)})
                                    continue
                            else:
                                # Default to NATM1
                                voice_id = "NATM1"
                                print(f"Initializing with persona='{persona[:50]}...', voice_id={voice_id} (default)")
                                voice_path, using_embedding = self._get_voice_prompt_path(voice_id)

                            # Import LMGen
                            from moshi.models import LMGen

                            # Create LMGen with proper parameters
                            lm_gen = LMGen(
                                self.lm,
                                audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                                sample_rate=self.mimi.sample_rate,
                                device=self.device,
                                frame_rate=self.mimi.frame_rate,
                            )

                            # Enable streaming mode
                            lm_gen.streaming_forever(1)

                            # Load voice prompt based on source
                            if embedding_data is not None:
                                # Load from inline embedding data
                                lm_gen.voice_prompt = "inline_embedding"
                                lm_gen.voice_prompt_audio = None
                                lm_gen.voice_prompt_embeddings = embedding_data["embeddings"].to(self.device)
                                lm_gen.voice_prompt_cache = embedding_data["cache"].to(self.device)
                            elif using_embedding and voice_path.endswith(".pt"):
                                # Load pre-computed embedding file
                                lm_gen.load_voice_prompt_embeddings(voice_path)
                            else:
                                # Load raw audio file (shouldn't happen with current setup)
                                lm_gen.load_voice_prompt(voice_path)

                            # Set text prompt (persona) via tokenizer
                            text_prompt = wrap_with_system_tags(persona)
                            lm_gen.text_prompt_tokens = self.text_tokenizer.encode(text_prompt)

                            # Reset Mimi streaming state for new conversation
                            # mimi: decodes output, other_mimi: encodes input
                            self.mimi.reset_streaming()
                            self.mimi.streaming_forever(1)
                            self.other_mimi.reset_streaming()
                            self.other_mimi.streaming_forever(1)

                            initialized = True

                            response_voice = voice_id if voice_id else "inline_embedding"
                            await ws.send_json({
                                "status": "ready",
                                "voice_id": response_voice,
                                "voice_type": "embedding" if using_embedding else "audio",
                                "sample_rate": SAMPLE_RATE,
                                "frame_samples": FRAME_SAMPLES,
                                "message": "Session initialized, send audio"
                            })
                            print(f"Session initialized successfully with voice={response_voice}")

                        except json.JSONDecodeError as e:
                            await ws.send_json({"error": f"Invalid JSON: {e}"})
                            continue
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            await ws.send_json({"error": f"Initialization failed: {e}"})
                            continue

                    elif "bytes" in message:
                        # Binary audio data (24kHz PCM int16)
                        if not initialized or lm_gen is None:
                            await ws.send_json({"error": "Send config first"})
                            continue

                        audio_bytes = message["bytes"]
                        input_buffer.extend(audio_bytes)

                        # Process complete frames
                        while len(input_buffer) >= BYTES_PER_FRAME:
                            frame_bytes = bytes(input_buffer[:BYTES_PER_FRAME])
                            del input_buffer[:BYTES_PER_FRAME]

                            # Convert bytes to tensor (int16 -> float32 normalized)
                            samples = np.frombuffer(frame_bytes, dtype=np.int16)
                            audio_tensor = torch.from_numpy(
                                samples.astype(np.float32) / 32768.0
                            )
                            # Shape: [batch=1, channels=1, samples]
                            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                            # Encode user audio to discrete codes (other_mimi)
                            with torch.no_grad():
                                codes = self.other_mimi.encode(audio_tensor)

                                # Process through LM - step returns output codes
                                # codes shape: [batch, codebooks, time]
                                for t in range(codes.shape[-1]):
                                    output_codes = lm_gen.step(codes[:, :, t:t+1])

                                    if output_codes is not None:
                                        # Decode output codes to audio
                                        output_audio = self.mimi.decode(output_codes)

                                        # Convert to PCM int16
                                        output_np = output_audio.squeeze().cpu().numpy()
                                        output_np = np.clip(output_np * 32768.0, -32768, 32767)
                                        output_bytes = output_np.astype(np.int16).tobytes()

                                        # Send audio response
                                        await ws.send_bytes(output_bytes)

            except WebSocketDisconnect:
                print("WebSocket disconnected")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"WebSocket error: {e}")
                try:
                    await ws.send_json({"error": str(e)})
                except Exception:
                    pass

        return api


@app.local_entrypoint()
def main():
    """Test the service locally."""
    print("PersonaPlex S2S service ready")
    print("Deploy with: modal deploy modal/personaplex_service.py")
    print(f"Available preset voices: {VOICE_IDS}")
    print("\nVoice Cloning Endpoints:")
    print("  POST /voice/encode - Encode audio to voice embedding (returns base64)")
    print("  POST /voice/clone  - Clone voice and save for reuse (returns voice_id)")
    print("  GET  /voices       - List all available voices")
    print("  DELETE /voice/{id} - Delete a custom cloned voice")
    print("\nWebSocket /ws accepts:")
    print("  - voice_id: Preset or custom cloned voice ID")
    print("  - voice_embedding: Base64-encoded embedding from /voice/encode")
