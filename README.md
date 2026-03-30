# Modal Inference Blueprint

A Tangle Blueprint for serving AI models on Modal GPUs. Operators earn revenue by hosting inference for text, voice, video, image, and music models.

116 models. 17 task types. Deploy in 5 minutes.

## Operator Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.80+)
- [Modal](https://modal.com/) account with GPU access
- [Foundry](https://book.getfoundry.sh/) (for contract interaction)
- `cargo-tangle` CLI: `cargo install cargo-tangle`

### 1. Deploy models to Modal

Pick which models you want to serve, then deploy them:

```bash
# See all 116 available models
python3 models/deploy.py list

# See costs before committing
python3 models/deploy.py cost --task tts

# Deploy specific models
python3 models/deploy.py deploy --model kokoro-tts --org your-modal-org
python3 models/deploy.py deploy --model hallo3 --org your-modal-org

# Or deploy an entire category
python3 models/deploy.py deploy --task tts --org your-modal-org
```

This generates Modal app files in `models/apps/` and runs `modal deploy` for each.

### 2. Generate your operator config

```bash
# Auto-generate operator.toml entries from your deployed models
python3 models/deploy.py gen-config --org your-modal-org --task tts >> config/operator.toml
```

Edit `config/operator.toml` — set your name, pricing, and billing preferences:

```toml
name = "My Inference Operator"

[server]
port = 8080

[cost]
idle_shutdown_minutes = 30    # Auto-stop idle Modal apps to save GPU cost

[[models]]
name = "kokoro-tts"
type = "tts"
modal_endpoint = "https://your-modal-org--inference-kokoro-tts-inference.modal.run"
price_per_1k = "0.001"
```

### 3. Build the operator

```bash
cargo build --release
```

Or use Docker:

```bash
docker build -t inference-operator .
docker run -v ./config:/app/config inference-operator
```

### 4. Deploy to Tangle

```bash
# Register the blueprint on Tangle
cargo tangle blueprint deploy

# Run the operator (connects to Tangle, starts HTTP server + heartbeat)
./target/release/modal-operator
```

For local testing without Tangle:

```bash
./target/release/standalone
```

### 5. Verify

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/models

# Test inference (TTS)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro-tts", "input": "Hello world", "voice": "af_heart"}'

# Check all deployed models
python3 models/deploy.py health --org your-modal-org
```

## Architecture

```
Developer ──→ Gateway ──→ Your Operator (this blueprint) ──→ Modal GPU
                │              │                                 │
                │         Axum HTTP server              Your deployed models
                │         x402 billing                  (kokoro, hallo3, etc.)
                │         Prometheus metrics
                │         Tangle heartbeat
                │
           Routes to the
           cheapest healthy
           operator for the
           requested model
```

## Model Categories

| Category | Models | Example |
|----------|--------|---------|
| Text LLMs | 34 | Llama 4, Qwen3, DeepSeek R1 |
| TTS | 22 | Kokoro, CosyVoice3, F5-TTS |
| STT | 7 | Whisper, SenseVoice, Parakeet |
| S2S | 3 | Moshi, Seamless M4T |
| Image | 12 | Flux, SDXL, SD3.5 |
| Video Generation | 11 | Wan2.1, CogVideoX, Mochi |
| Video Avatar | 5 | Hallo3, LivePortrait |
| Video Lipsync | 3 | LatentSync, MuseTalk |
| Video Understanding | 4 | InternVideo2.5, Qwen VL |
| Music | 4 | MusicGen, Stable Audio |
| Diarization | 2 | Pyannote, NeMo |
| Other | 9 | VAD, LID, RVC, DeepFilterNet |

36 models have production inference code in `models/services/`. The rest are auto-generated from Jinja2 templates.

## Billing

Operators set their own prices. Two modes:

**x402 / ShieldedCredits (on-chain):**
Developers sign a `SpendAuth` (EIP-712). The operator validates on-chain, serves inference, then claims payment. No intermediary.

```toml
[billing]
required = true

[billing.pricing]
price_per_1k_chars = 5000              # TTS: 0.005 tsUSD per 1K chars
price_per_second_audio = 400           # STT: 0.0004 tsUSD per second
price_per_second_video = 40000         # Video: 0.04 tsUSD per second
price_per_million_input_tokens = 300   # LLMs: 0.0003 tsUSD per M input tokens
```

**No billing (free / standalone):**
Set `billing.required = false`. All requests served without payment validation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/speech` | OpenAI-compatible TTS |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible STT |
| GET | `/v1/audio/models` | List models |
| POST | `/proxy/:model/*path` | Raw proxy to any model endpoint |
| GET | `/health` | Health check (all models) |
| GET | `/metrics` | Prometheus metrics |

## Smart Contract

`InferenceBSM.sol` — on-chain operator registration, metrics, and slashing.

```bash
cd contracts
forge build
forge test -vvv

# Deploy
TSUSD_ADDRESS=0x... ADMIN_ADDRESS=0x... forge script script/Deploy.s.sol --broadcast
```

## CLI Reference

```bash
# Models
python3 models/deploy.py list                          # All 116 models
python3 models/deploy.py list --task video-avatar      # Filter by task
python3 models/deploy.py cost --task tts               # GPU cost estimates
python3 models/deploy.py deploy --model X --org Y      # Deploy one model
python3 models/deploy.py deploy --task tts --org Y     # Deploy all TTS
python3 models/deploy.py gen-config --org Y            # Generate operator.toml
python3 models/deploy.py health --org Y                # Health check all

# Operator
cargo build --release                  # Build
cargo test                             # Test
cargo tangle blueprint deploy          # Register on Tangle
./target/release/standalone            # Run standalone (no Tangle)
./target/release/modal-operator        # Run with Tangle
```

## License

MIT OR Apache-2.0
