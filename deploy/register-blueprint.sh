#!/usr/bin/env bash
# Register the Modal multi-modal inference blueprint on Tangle.
#
# Single-shot flow: deploys InferenceBSM (constructor-initialized with the
# payment token + admin) AND calls Tangle.createBlueprint in the same
# broadcast via `contracts/script/RegisterBlueprint.s.sol`.
#
# Prerequisites:
#   - forge installed
#   - Deployer wallet funded on the target network
#
# Usage (Base Sepolia, against the already-deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export PAYMENT_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia
#   # Optional — defaults to the deployer address:
#   # export ADMIN_ADDRESS=0xYourAdminSafeOrEOA
#   ./deploy/register-blueprint.sh
#
# Local anvil (LocalTestnet snapshot):
#
#   export RPC_URL=http://127.0.0.1:8545
#   ./deploy/register-blueprint.sh   # uses anvil deployer key + Tangle/USDC defaults
#
# Optional operator-registration helper overrides (used to emit a sample
# `registerOperator` calldata for the user — no on-chain effect):
#   MODELS             Comma-separated model names (default: kokoro-tts,whisper-large-v3,sdxl)
#   TASK_TYPES         Comma-separated task types  (default: tts,stt,image)
#   TOTAL_VRAM         Total VRAM in MiB           (default: 80000)
#   GPU_MODEL          GPU model string            (default: NVIDIA H100)
#   ENDPOINT           Operator HTTP endpoint      (default: https://your-operator.example.com)
#
# Outputs (parsed by deployment scripts, do not change without coordinating):
#   DEPLOY_INFERENCE_BSM=<address>
#   DEPLOY_INFERENCE_BSM_ADMIN=<address>
#   DEPLOY_INFERENCE_PAYMENT_TOKEN=<address>
#   DEPLOY_INFERENCE_BLUEPRINT_ID=<u64>

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"

MODELS="${MODELS:-kokoro-tts,whisper-large-v3,sdxl}"
TASK_TYPES="${TASK_TYPES:-tts,stt,image}"
TOTAL_VRAM="${TOTAL_VRAM:-80000}"
GPU_MODEL="${GPU_MODEL:-NVIDIA H100}"
ENDPOINT="${ENDPOINT:-https://your-operator.example.com}"

echo "=== Modal Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:    $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core: ${TANGLE_CORE:-<default from RegisterBlueprint.s.sol>}"
echo "Payment:     ${PAYMENT_TOKEN:-<default USDC sepolia>}"
echo "Admin:       ${ADMIN_ADDRESS:-<defaults to deployer>}"
echo "Models:      $MODELS"
echo "Tasks:       $TASK_TYPES"
echo "GPU:         $GPU_MODEL ($TOTAL_VRAM MiB)"
echo "Endpoint:    $ENDPOINT"
echo ""

cd "$(dirname "$0")/../contracts"

# Deploy BSM AND register the blueprint in one forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    PAYMENT_TOKEN="${PAYMENT_TOKEN:-}" \
    ADMIN_ADDRESS="${ADMIN_ADDRESS:-}" \
    forge script script/RegisterBlueprint.s.sol \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract the BSM address + blueprint ID for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_INFERENCE_BSM=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_INFERENCE_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output"
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID:    $BLUEPRINT_ID"
echo "InferenceBSM:    $BSM_ADDRESS"
echo ""

# Build a sample operator-registration calldata. InferenceBSM.onRegister
# expects `(string[] models, string[] taskTypes, uint32 gpuVramMib,
# string gpuModel, string endpoint)`. We turn the comma-separated env
# inputs into ABI string arrays via cast.
MODELS_JSON="[$(echo "$MODELS"      | awk -F, '{for(i=1;i<=NF;i++) printf "%s\"%s\"", (i>1?",":""), $i}')]"
TASKS_JSON="[$(echo "$TASK_TYPES" | awk -F, '{for(i=1;i<=NF;i++) printf "%s\"%s\"", (i>1?",":""), $i}')]"

REG_INPUTS=$(cast abi-encode \
    "f(string[],string[],uint32,string,string)" \
    "$MODELS_JSON" "$TASKS_JSON" "$TOTAL_VRAM" "$GPU_MODEL" "$ENDPOINT")

echo "Operator registration inputs (use these to register an operator):"
echo "  $REG_INPUTS"
echo ""
echo "To register an operator now:"
echo "  cast send ${TANGLE_CORE:-<TANGLE_CORE>} \\"
echo "    'registerOperator(uint64,bytes)' $BLUEPRINT_ID $REG_INPUTS \\"
echo "    --rpc-url $RPC_URL --private-key \$OPERATOR_KEY"
