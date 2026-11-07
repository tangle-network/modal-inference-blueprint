# Modal Inference Blueprint

Use [README.md](README.md) for setup and Beelink gates; [Cargo.toml](Cargo.toml) owns the SDK pin and its source defines supported APIs.

Keep GPU allocation and retirement in the SDK remote-provider adapter; this blueprint serves HTTP.
Verify server changes through real HTTP, substituting only external providers.
Verify chain jobs through the production runner and recorded results; deploy actual contracts to check affected registration, pricing, payment and access rules.
Report provider substitutes and unavailable prerequisites.
