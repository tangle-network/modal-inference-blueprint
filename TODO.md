# modal-inference-blueprint — Status

## Done
- [x] Rust workspace + operator crate (lib + bin pattern)
- [x] BlueprintRunner wiring (main.rs)
- [x] Router with generic InferenceJob (lib.rs)
- [x] Axum HTTP server with OpenAI-compatible endpoints (server.rs)
- [x] Modal proxy — routes requests to operator's Modal endpoints (proxy.rs)
- [x] Prometheus metrics + on-chain metric pairs (metrics.rs)
- [x] Auto-registration with gateway marketplace (registry.rs)
- [x] TOML config for models, pricing, QoS (config.rs)
- [x] InferenceBSM.sol — multi-modal contract with PricingUnit enum
- [x] SlashingLib integration (propose → dispute → execute/cancel)
- [x] Admin controls (toggle slashing, permitted callers, model config)
- [x] Heartbeat config (50 blocks, 3 missed threshold)
- [x] Foundry project with tnt-core 0.10.4 + OpenZeppelin 5.1.0
- [x] Compiles clean against blueprint-sdk main branch
- [x] Model registry — 116 models across 17 task types
- [x] 38 production service files (from Modal inference prototypes) in models/services/
- [x] 11 Jinja2 engine templates for auto-generated models
- [x] deploy.py CLI — deploy, list, gen-config, health, cost
- [x] x402/ShieldedCredits billing (authorizeSpend, claimPayment, EIP-712)
- [x] Per-task-type pricing for all 17 task types
- [x] Idle modal app management (auto-stop, wake on request)
- [x] Background health check loop (probes all Modal endpoints)
- [x] Forge tests for InferenceBSM (model config, pricing units, admin, slashing)
- [x] Forge deploy script
- [x] Dockerfile for standalone operator
- [x] CI: cargo clippy + forge test + registry validation

## Remaining
- [ ] Verify contract against Tangle testnet
- [ ] Load test: 50 concurrent requests through proxy
- [ ] Operator onboarding guide (step-by-step docs)
