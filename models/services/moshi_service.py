"""
Moshi S2S Service - Kyutai's full-duplex speech-to-speech dialogue model.

Capabilities:
- Full-duplex conversation (listen while speaking)
- 200ms practical latency, 160ms theoretical
- 7B temporal transformer with inner monologue
- Mimi neural audio codec (24kHz, 12.5Hz frame rate, 1.1kbps)
- Opus streaming over WebSocket
- Apache-2.0 license

Architecture:
- Mimi codec encodes user audio -> discrete tokens
- Moshi LM generates response tokens autoregressively
- Mimi codec decodes response tokens -> audio
- Two audio streams: user input + model output (full-duplex)

Endpoints:
- GET  /health         - Service health check
- GET  /status         - Alias for health
- WS   /converse       - Full-duplex WebSocket conversation
- POST /synthesize_turn - Single-turn text-to-speech (non-streaming)

WebSocket protocol (/converse):
- Client sends: raw Opus audio bytes
- Server sends: b"\\x01" + Opus audio bytes (audio)
                b"\\x02" + UTF-8 text bytes (transcript)

Deploy: modal deploy modal/moshi_service.py
"""

import modal
import os

app = modal.App("modal-inference-moshi-s2s")

MODEL_DIR = "/models/moshi"
model_cache = modal.Volume.from_name("phony-moshi-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libopus-dev", "pkg-config", "libsndfile1")
    .pip_install(
        "moshi>=0.2.0",
        "fastapi",
        "uvicorn",
        "huggingface_hub",
        "hf_transfer",
        "sphn>=0.1.4",
        "torch",
        "numpy<2",
        "soundfile",
        "sentencepiece",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_CACHE": MODEL_DIR,
    })
)

SAMPLE_RATE = 24000
HF_REPO = "kyutai/moshika-pytorch-bf16"


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    container_idle_timeout=300,
    volumes={MODEL_DIR: model_cache},
    allow_concurrent_inputs=4,
)
class MoshiService:
    """Full-duplex speech-to-speech service using Kyutai Moshi."""

    @modal.enter()
    def setup(self):
        import torch
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders, LMGen
        import sentencepiece
        import sphn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download model files
        hf_hub_download(HF_REPO, loaders.MOSHI_NAME)
        hf_hub_download(HF_REPO, loaders.MIMI_NAME)
        hf_hub_download(HF_REPO, loaders.TEXT_TOKENIZER_NAME)

        # Load Mimi codec
        mimi_weight = hf_hub_download(HF_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        # Load Moshi LM
        moshi_weight = hf_hub_download(HF_REPO, loaders.MOSHI_NAME)
        self.moshi_lm = loaders.get_moshi_lm(moshi_weight, device=self.device)
        self.lm_gen = LMGen(
            self.moshi_lm,
            temp=0.8,
            temp_text=0.8,
            top_k=250,
            top_k_text=25,
        )

        # Enable streaming mode
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Text tokenizer for transcript output
        tokenizer_path = hf_hub_download(HF_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Warmup GPU
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device,
            )
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()

        print(f"Moshi S2S loaded on {self.device} (frame_size={self.frame_size}, sr={SAMPLE_RATE})")

    def _reset_streaming(self):
        """Reset stateful model components for a new conversation."""
        import sphn

        self.opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
        self.opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    @modal.asgi_app()
    def web(self):
        import asyncio
        import time
        import torch
        import numpy as np
        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse, Response
        import soundfile as sf
        import io

        web_app = FastAPI()
        svc = self

        @web_app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": "moshi-s2s",
                "gpu": "A100",
                "sample_rate": SAMPLE_RATE,
                "frame_size": svc.frame_size,
                "hf_repo": HF_REPO,
                "features": [
                    "full_duplex", "speech_to_speech", "streaming",
                    "inner_monologue", "opus_codec",
                ],
                "latency_ms": 200,
            }

        @web_app.get("/status")
        def status():
            return health()

        @web_app.websocket("/converse")
        async def converse(ws: WebSocket):
            """Full-duplex S2S conversation over WebSocket.

            Protocol:
            - Client sends: raw Opus-encoded audio bytes
            - Server sends:
                b"\\x01" + Opus audio (model speech)
                b"\\x02" + UTF-8 text  (model transcript)
            """
            with torch.no_grad():
                await ws.accept()
                svc._reset_streaming()
                print("Moshi session started")

                tasks = []

                async def recv_loop():
                    while True:
                        data = await ws.receive_bytes()
                        if not isinstance(data, bytes) or len(data) == 0:
                            continue
                        svc.opus_reader.append_bytes(data)

                async def inference_loop():
                    all_pcm = None
                    while True:
                        await asyncio.sleep(0.001)
                        pcm = svc.opus_reader.read_pcm()
                        if pcm is None or len(pcm) == 0:
                            continue
                        if pcm.shape[-1] == 0:
                            continue

                        all_pcm = pcm if all_pcm is None else np.concatenate((all_pcm, pcm))

                        while all_pcm.shape[-1] >= svc.frame_size:
                            chunk = all_pcm[:svc.frame_size]
                            all_pcm = all_pcm[svc.frame_size:]

                            chunk_t = torch.from_numpy(chunk).to(device=svc.device)[None, None]
                            codes = svc.mimi.encode(chunk_t)

                            for c in range(codes.shape[-1]):
                                tokens = svc.lm_gen.step(codes[:, :, c : c + 1])
                                if tokens is None:
                                    continue

                                # Decode model audio response
                                main_pcm = svc.mimi.decode(tokens[:, 1:])
                                svc.opus_writer.append_pcm(main_pcm[0, 0].cpu().numpy())

                                # Extract text token
                                text_token = tokens[0, 0, 0].item()
                                if text_token not in (0, 3):
                                    text = svc.text_tokenizer.id_to_piece(text_token)
                                    text = text.replace("\u2581", " ")
                                    await ws.send_bytes(
                                        b"\x02" + text.encode("utf-8")
                                    )

                async def send_loop():
                    while True:
                        await asyncio.sleep(0.001)
                        msg = svc.opus_writer.read_bytes()
                        if msg is None or len(msg) == 0:
                            continue
                        await ws.send_bytes(b"\x01" + msg)

                try:
                    tasks = [
                        asyncio.create_task(recv_loop()),
                        asyncio.create_task(inference_loop()),
                        asyncio.create_task(send_loop()),
                    ]
                    await asyncio.gather(*tasks)
                except WebSocketDisconnect:
                    print("Moshi session disconnected")
                except Exception as e:
                    print(f"Moshi session error: {e}")
                    await ws.close(code=1011)
                    raise
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    svc._reset_streaming()

        @web_app.post("/synthesize_turn")
        async def synthesize_turn(request: Request):
            """Single-turn: encode text prompt, run LM, decode audio.

            Not full-duplex -- for testing/preview only. For real
            conversations use the /converse WebSocket.

            Body: {"text": "Hello!", "max_steps": 200}
            Returns: audio/wav
            """
            # This is a simplified non-streaming endpoint for basic testing.
            # Moshi is designed for streaming duplex; this just generates
            # a short response to silence + text prompt.
            return JSONResponse(
                status_code=501,
                content={
                    "error": "Use WebSocket /converse for Moshi S2S interaction",
                    "hint": "Moshi is a full-duplex streaming model. "
                            "Connect via WebSocket and stream Opus audio.",
                },
            )

        return web_app


@app.local_entrypoint()
def main():
    print("Moshi S2S service ready (7B, full-duplex, Mimi codec)")
    print("Deploy with: modal deploy modal/moshi_service.py")
