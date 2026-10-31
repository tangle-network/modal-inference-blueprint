// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import { Script, console2 } from "forge-std/Script.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { InferenceBSM } from "../src/InferenceBSM.sol";
/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}
/// @notice Deploys the non-upgradeable InferenceBSM and registers it. Tempo uses DeployTempo.
contract RegisterBlueprint is Script {
    uint256 constant DEFAULT_DEPLOYER_KEY = 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;
    address constant DEFAULT_PAYMENT_TOKEN = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;
    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address paymentToken = vm.envOr("PAYMENT_TOKEN", DEFAULT_PAYMENT_TOKEN);
        address admin = vm.envOr("ADMIN_ADDRESS", vm.addr(deployerKey));
        ITangle tangle = ITangle(tangleAddr);
        // eventRate below is 0, which makes the job free. Tempo pathUSD uses DeployTempo.
        require(
            tangleAddr != 0xff137B9C879c47C28ce389e84501925438AB4cdA, "Tempo: use contracts/script/DeployTempo.s.sol"
        );
        vm.startBroadcast(deployerKey);
        InferenceBSM bsm = new InferenceBSM(paymentToken, admin);
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));
        vm.stopBroadcast();
        console2.log("DEPLOY_INFERENCE_BSM=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_INFERENCE_BSM_ADMIN=%s", vm.toString(admin));
        console2.log("DEPLOY_INFERENCE_PAYMENT_TOKEN=%s", vm.toString(paymentToken));
        console2.log("DEPLOY_INFERENCE_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }
    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/modal-inference-blueprint";
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0, // unbounded
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // free on this path. Tempo floor is DeployTempo. Per-model prices stay in the operator.
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
        jobs[0] = Types.JobDefinition({
            name: "inference",
            description: "Run multi-modal inference via Modal GPUs (chat/TTS/STT/image/video/music)",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}
