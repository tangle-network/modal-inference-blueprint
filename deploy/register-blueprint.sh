#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$(cargo metadata --locked --manifest-path "$ROOT/Cargo.toml" --format-version 1 | jq -er '.packages[] | select(.name == "tangle-inference-core") | .manifest_path | sub("/Cargo.toml$"; "")')"
BLUEPRINT_DEFINITION="$(node "$CORE_DIR/scripts/register-blueprint.mjs" "$ROOT/deploy/definition.json" "$ROOT")"
export BLUEPRINT_DEFINITION
trap 'rm -f -- "$BLUEPRINT_DEFINITION"' EXIT
REGISTRATION_SCRIPT=script/Deploy.s.sol:DeployInference
REGISTRATION_CORE="${TANGLE_CORE:-}"
if [[ "$(cast chain-id --rpc-url "${RPC_URL:?Set RPC_URL}")" == 42431 || "${REGISTRATION_CORE,,}" == 0xff137b9c879c47c28ce389e84501925438ab4cda ]]; then
    REGISTRATION_SCRIPT=script/DeployTempo.s.sol:DeployTempo
fi
cd "$ROOT/contracts"
forge script "$REGISTRATION_SCRIPT" --remappings "tangle-inference-core/=$CORE_DIR/contracts/" --rpc-url "$RPC_URL" --broadcast --slow
