// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { InferenceBSM } from "../src/InferenceBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys the (non-upgradeable) InferenceBSM and registers the
///         modal-inference blueprint on Tangle in a single broadcast.
/// @dev    Run via:
///           forge script contracts/script/RegisterBlueprint.s.sol \
///             --rpc-url $RPC_URL --broadcast --slow
///
///         InferenceBSM is a plain constructor-initialized contract
///         (tsUSD, admin); there is no proxy. This mirrors the modal
///         blueprint's deployment manifest while reusing the
///         Pattern-B `Tangle.createBlueprint` flow proven for vLLM.
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real
    // chains (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // USDC on Base Sepolia. The modal operator settles in this token under the
    // shielded billing flow. For other networks pass PAYMENT_TOKEN via env.
    address constant DEFAULT_PAYMENT_TOKEN = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address paymentToken = vm.envOr("PAYMENT_TOKEN", DEFAULT_PAYMENT_TOKEN);
        // BSM admin defaults to the deployer; passing ADMIN_ADDRESS overrides it
        // (e.g. for hand-off to a Safe / governance multisig).
        address admin = vm.envOr("ADMIN_ADDRESS", vm.addr(deployerKey));

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy InferenceBSM (constructor-initialized, non-upgradeable) ──
        InferenceBSM bsm = new InferenceBSM(paymentToken, admin);

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_INFERENCE_BSM=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_INFERENCE_BSM_ADMIN=%s", vm.toString(admin));
        console2.log("DEPLOY_INFERENCE_PAYMENT_TOKEN=%s", vm.toString(paymentToken));
        console2.log("DEPLOY_INFERENCE_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/modal-inference-blueprint";
        // metadataHash is a digest of the canonical metadata JSON. Until that
        // payload is pinned via IPFS, derive it from the metadataUri so the
        // value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Event-driven pricing: operators are paid per inference call rather
        // than on a fixed subscription cadence. The Rust operator advertises
        // a single `INFERENCE_JOB` (id 0) whose `model` field selects the
        // Modal endpoint — chat, TTS, STT, image, video, music — and the
        // BSM enforces per-model `PricingUnit` rates set by the admin.
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0, // unbounded
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // per-model rates live in InferenceBSM.modelConfigs
        });

        def.metadata = Types.BlueprintMetadata({
            name: "Modal Inference Blueprint",
            description: "Modal-backed multi-modal inference operator (chat, TTS, STT, image, video, music) with shielded billing",
            author: "Tangle Network",
            category: "AI/Inference",
            codeRepository: "https://github.com/tangle-network/modal-inference-blueprint",
            logo: "",
            website: "https://tangle.tools",
            license: "MIT OR Apache-2.0",
            profilingData: ""
        });

        def.jobs = _buildJobs();

        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "modal-inference-blueprint",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/modal-inference-blueprint",
                "./target/release/modal-inference-blueprint"
            ),
            testing: Types.TestingSource("modal-inference-blueprint", "modal-inference-blueprint", "."),
            binaries: bins
        });

        def.supportedMemberships = new Types.MembershipModel[](1);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
    }

    function _buildJobs() internal pure returns (Types.JobDefinition[] memory jobs) {
        jobs = new Types.JobDefinition[](1);
        // Job 0: inference (matches `INFERENCE_JOB` in operator/src/lib.rs)
        //   inputs:  (string model, bytes inputData, string inputType, string outputType)
        //   outputs: (bytes outputData, uint32 unitsConsumed, string outputType, uint32 latencyMs)
        // Real traffic flows through the OpenAI-compatible HTTP proxy; the
        // on-chain job exists as a billing-verification path. Params/result
        // shapes are enforced by the Rust operator. On-chain schemas are kept
        // empty to match the pattern used by the vLLM inference blueprint —
        // hex-encoded schemas via tnt-core's SchemaLib can be added once that
        // surface stabilizes.
        jobs[0] = Types.JobDefinition({
            name: "inference",
            description: "Run multi-modal inference via Modal GPUs (chat/TTS/STT/image/video/music)",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}
