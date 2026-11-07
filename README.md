# Modal inference blueprint

Serve OpenAI and task-specific HTTP requests through an endpoint allocated by `blueprint-sdk` remote providers. The SDK manages GPU allocation, credentials, prices and retirement.

Copy `config/example.toml` to `config/operator.toml` and set the network and billing contract. Set `[[modal.models]].modal_endpoint` and the `[gpu]` class, count and VRAM per GPU from the SDK allocation receipt. The host runs an HTTP proxy; its own GPU inventory is not the remote allocation. The example omits credentials; core's SDK-keystore billing-wallet integration remains required for production startup. Use `billing_required = false` only for a deliberately unbilled local endpoint.

Run `cargo run --bin standalone` for HTTP serving or `cargo run --bin modal-operator` for chain jobs. Chain jobs preserve `(model, inputData, inputType, outputType)` and call the configured provider through the same model registry as HTTP. Text is adapted to the model task, JSON is forwarded as a provider request, and audio is forwarded as WAV. The existing five-field operator registration wire remains supported; SDK-native GPU count/class requirements follow the D3 release. `/health` checks the configured backends, `/metrics` exports core metrics, and `/v1/audio/speech`, `/v1/audio/transcriptions`, and `/proxy/:model/*path` serve inference.

Existing SDK-allocated endpoints can be served while the SDK Modal adapter awaits release. Select serving images and model-family flags from the core recipe catalog.

Deploy the manager with [Deploy.s.sol](contracts/script/Deploy.s.sol), `TANGLE_CORE`, `PAYMENT_TOKEN`, `ADMIN_ADDRESS` and a Foundry keystore. Set its returned address in [definition.json](deploy/definition.json), then follow the [cargo-tangle deployment guide](https://github.com/tangle-network/blueprint#deploying-to-testnetmainnet) using the same wallet address in the SDK keystore. Supply the verified release tag and executable digest through the CLI artifact options. After registration, [DeployTempo.s.sol](contracts/script/DeployTempo.s.sol) pins settlement for the supplied `BLUEPRINT_ID`; it uses the selected Foundry signer. Native archives and `dist-manifest.json` are produced by [cargo-dist](dist-workspace.toml) through [Release](.github/workflows/release.yml).

`cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cd contracts && forge test` are the repository gates, run through `beelink-gate`.
