# Modal Inference Blueprint

Use [README.md](README.md) for setup and [Cargo.toml](Cargo.toml) for the selected Blueprint SDK.
Read the selected SDK source for supported test APIs instead of copying dependency or API examples into this guide.

## Verification

Use the Beelink checks in [README.md](README.md) for Rust and contracts.
Allocate and retire GPU capacity through the SDK remote-provider adapter; keep this blueprint as an HTTP adapter.
For server changes, send requests to the actual server and replace only external providers where needed.
For chain integration, verify job submission, operator processing, and result recording through the production runner.
Exercise contract changes with actual deployments, including registration, pricing, payment, and access rules as affected.
Test nontrivial logic and error paths; avoid checks that succeed without exercising the required behavior.
Report provider substitutes and unavailable prerequisites with the result.
