# modal-inference-blueprint — Remaining Work

## Done
- [x] Rust workspace + operator crate (lib + bin pattern)
- [x] BlueprintRunner wiring (main.rs)
- [x] Router with generic InferenceJob (lib.rs)
- [x] Axum HTTP server with OpenAI-compatible endpoints (server.rs)
- [x] Modal proxy — routes requests to operator's Modal endpoints (proxy.rs)
- [x] Prometheus metrics + on-chain metric pairs (metrics.rs)
- [x] Auto-registration with gateway marketplace (registry.rs)
- [x] TOML config for models, pricing, QoS (config.rs)
- [x] InferenceBSM.sol — inherits BlueprintServiceManagerBase
- [x] SlashingLib integration (propose → dispute → execute/cancel)
- [x] Admin controls (toggle slashing, permitted callers, model config)
- [x] Heartbeat config (50 blocks, 3 missed threshold)
- [x] Foundry project with tnt-core 0.10.4 + OpenZeppelin 5.1.0
- [x] Compiles clean against blueprint-sdk main branch
- [x] Model registry — 24 video + 10 TTS + 4 STT + text/image/music models

## Remaining

### Blueprint Core
- [ ] Model engine — `git clone` modal-apps repo on start, `modal deploy` per model
- [ ] Lifecycle management — `modal app stop` on service termination
- [ ] Auto-update — periodic `git pull` + redeploy if scripts changed
- [ ] Health check loop — poll Modal endpoints, report via QoS heartbeat
- [ ] On-chain metrics submission — call `submitMetrics()` on BSM periodically

### Contract
- [ ] Forge tests for InferenceBSM
- [ ] Test slashing flow: propose → dispute window → execute
- [ ] Test auto-suspend on low uptime
- [ ] Test admin controls (toggle slashing, permitted callers)
- [ ] Deploy script (forge script)
- [ ] Verify against Tangle testnet

### Infrastructure
- [ ] CI: cargo test + cargo clippy + forge test
- [ ] Docker image for operators who don't want to build from source
- [ ] Example configs for popular models (CosyVoice, Whisper, Hallo3)

### Integration with Gateway
- [ ] Gateway scrapes operator `/metrics` endpoint
- [ ] Gateway reads on-chain operator reputation from BSM
- [ ] Gateway routes to operators based on reputation score
- [ ] Gateway shows operator status on marketplace pages
- [ ] Operator dashboard reads from BSM (on-chain metrics)

### Documentation
- [ ] Operator onboarding guide (step-by-step)
- [ ] Earnings calculator
- [ ] Model selection guide (which models have demand)
- [ ] Troubleshooting (common deploy issues)
