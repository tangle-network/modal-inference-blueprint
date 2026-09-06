# Modal Inference Blueprint

Use [README.md](README.md) for setup and [Cargo.toml](Cargo.toml) for the selected Blueprint SDK.
Read the selected SDK source for supported test APIs instead of copying dependency or API examples into this guide.

## Verification

Follow [.github/workflows/ci.yml](.github/workflows/ci.yml) for Rust, contract, and model-registry checks.
Registry changes must parse, generate deployable applications, and pass syntax checks through the existing dry-run path.
For server changes, send requests to the actual server and replace only external providers where needed.
For chain integration, verify job submission, operator processing, and result recording through the production runner.
Exercise contract changes with actual deployments, including registration, pricing, payment, and access rules as affected.
Test nontrivial logic and error paths; avoid checks that succeed without exercising the required behavior.
Report provider substitutes and unavailable prerequisites with the result.
