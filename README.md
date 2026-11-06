# Modal inference blueprint

The operator adapts OpenAI and task-specific HTTP requests to an endpoint allocated by `blueprint-sdk` remote providers. GPU allocation, credentials, prices, lifetime and retirement belong to the SDK. The blueprint runs no Modal CLI, deploys no Modal apps and owns no GPU price table.

Copy `config/example.toml` to `config/operator.toml`, set the network and billing contract, and provide the signing key through `MODAL_OP_TANGLE__OPERATOR_KEY`. Set `[[modal.models]].modal_endpoint` and the `[gpu]` class, count and VRAM per GPU from the SDK allocation receipt. The host runs an HTTP proxy; its own GPU inventory is not the remote allocation. The example's blank key is intentional: it must be supplied before startup. Use `billing_required = false` only for a deliberately unbilled local endpoint.

Run `cargo run --bin standalone` for HTTP serving or `cargo run --bin modal-operator` for chain jobs. The same model endpoint and per-task billing rules serve both. `/health` checks the configured backends, `/metrics` exports core metrics, and `/v1/audio/speech`, `/v1/audio/transcriptions`, and `/proxy/:model/*path` serve inference.

The SDK lane owns the Modal adapter. Until that adapter is released, allocation through Modal remains unavailable; an existing provider-allocated endpoint can still be served. There is no fallback rental path in this repository. Model-family serving images and flags are owned by the core recipe catalog.

`cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cd contracts && forge test` are the repository gates, run through `beelink-gate`. Releases use `.github/workflows/release.yml`.
