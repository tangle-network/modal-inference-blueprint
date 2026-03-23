# Modal Inference Blueprint

A Tangle Blueprint for serving voice AI models through the ph0ny Gateway. Operators deploy Modal apps (TTS, STT, diarization, etc.) and run this blueprint to earn revenue.

## What This Does

```
You (operator) deploy:
1. Modal apps (any of the 32 ph0ny voice model scripts)
2. This blueprint (Rust binary)

The blueprint handles:
- Tangle network registration + heartbeat
- OpenAI-compatible HTTP proxy
- Prometheus metrics + on-chain reputation
- Auto-registration with ph0ny marketplace
- Revenue: you earn 80%, ph0ny takes 20%
```

## Quick Start

```bash
# 1. Deploy a Modal voice model
cd /path/to/ph0ny/infra/ingest/modal
modal deploy cosyvoice3_service.py

# 2. Clone this blueprint
git clone https://github.com/ph0ny/modal-inference-blueprint
cd modal-inference-blueprint

# 3. Configure
cp config/example.toml config/operator.toml
# Edit: set your Modal endpoint URL, pricing, payout info

# 4. Build
cargo build --release

# 5. Deploy to Tangle
cargo tangle blueprint deploy

# 6. Run
./target/release/modal-operator
```

## Architecture

```
                        ┌─────────────────────────┐
Developer request ────→ │    ph0ny Gateway         │
                        │    api.ph0ny.com          │
                        └──────────┬──────────────┘
                                   │ routes to operator
                                   ▼
                        ┌─────────────────────────┐
                        │  This Blueprint (Rust)   │
                        │  - Auth + billing        │
                        │  - Metrics + heartbeat   │
                        │  - HTTP proxy            │
                        └──────────┬──────────────┘
                                   │ proxies to Modal
                                   ▼
                        ┌─────────────────────────┐
                        │  Your Modal Deployment   │
                        │  cosyvoice3.modal.run    │
                        │  (runs on YOUR GPU)      │
                        └─────────────────────────┘
```

## Supported Models

Any of the 32 ph0ny Modal scripts:

| Category | Models |
|----------|--------|
| TTS | CosyVoice 3, Fish S2 Pro, Chatterbox, Kokoro, F5-TTS, Orpheus, IndexTTS-2, TADA, Higgs Audio, VibeVoice, Dia, Sesame CSM, Qwen3-TTS, Spark-TTS, OpenVoice |
| STT | Distil-Whisper, Parakeet, SenseVoice |
| S2S | Moshi, Step-Audio 2 |
| Diarization | Pyannote 3.1, NeMo Sortformer |
| Enhancement | DeepFilterNet |
| Translation | SeamlessM4T v2 |
| Language ID | SpeechBrain LID |
| Speaker ID | (via Pyannote embeddings) |
| VAD | Silero VAD v5 |
| Voice Conversion | RVC v2 |

## Configuration

See `config/example.toml`. Key fields:

```toml
name = "My Voice Operator"

[[models]]
name = "cosyvoice3"       # Model identifier
type = "tts"              # Task type
modal_endpoint = "https://your-org--cosyvoice3-service.modal.run"
price_per_1k = "0.005"    # Your price per 1K characters (you set this)
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/audio/speech | OpenAI-compatible TTS |
| POST | /v1/audio/transcriptions | OpenAI-compatible STT |
| GET | /v1/audio/models | List available models |
| POST | /proxy/:model/*path | Raw proxy to any model |
| GET | /health | Health check (all models) |
| GET | /metrics | Prometheus metrics |

## Reputation

The blueprint sends heartbeats to Tangle every 30 seconds and submits metrics on-chain. Your reputation score determines how much traffic you receive:

- **Uptime** (35%): heartbeats received / expected
- **Quality** (25%): MOS score (TTS) or WER (STT) from benchmarks
- **Latency** (20%): p50 vs model class average
- **Reliability** (10%): success / total requests
- **Volume** (10%): log-scale total requests served

## Revenue

- Developers pay ph0ny (credits or Stripe)
- ph0ny routes requests to you
- You earn **80%** of the per-request revenue
- ph0ny takes **20%** platform fee
- Monthly payout via Stripe Connect or on-chain

## SDK Crates Used

- `blueprint-sdk`: Runner, router, job handlers
- `blueprint-qos`: Heartbeat, metrics, Prometheus
- `blueprint-pricing-engine`: Operator pricing
- `prometheus`: Metrics export

## Build

```bash
cargo build --release
cargo test
```

## License

MIT OR Apache-2.0
