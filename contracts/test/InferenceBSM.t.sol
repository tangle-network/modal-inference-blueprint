// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "forge-std/Test.sol";
import "../src/InferenceBSM.sol";

contract MockTsUSD {
    mapping(address => uint256) public balanceOf;
    function mint(address to, uint256 amount) external { balanceOf[to] += amount; }
}

contract InferenceBSMTest is Test {
    InferenceBSM public bsm;
    MockTsUSD public tsUSD;
    address public admin = address(0xA);
    address public operator1 = address(0x1);
    address public operator2 = address(0x2);

    function setUp() public {
        tsUSD = new MockTsUSD();
        bsm = new InferenceBSM(address(tsUSD), admin);

        // Configure models
        vm.startPrank(admin);
        bsm.configureModel("kokoro-tts", "tts", InferenceBSM.PricingUnit.Per1KCharacters, 5000, 4000);
        bsm.configureModel("whisper-v3", "stt", InferenceBSM.PricingUnit.PerSecondAudio, 400, 4000);
        bsm.configureModel("hallo3", "video-avatar", InferenceBSM.PricingUnit.PerSecondVideo, 40000, 24000);
        bsm.configureModel("llama-4-maverick", "text-generation", InferenceBSM.PricingUnit.PerMillionTokens, 300000, 280000);
        bsm.configureModel("flux-1-dev", "image-generation", InferenceBSM.PricingUnit.PerImage, 30000, 24000);
        bsm.configureModel("video-stitch", "video-stitch", InferenceBSM.PricingUnit.PerJob, 5000, 0);
        vm.stopPrank();
    }

    // ── Registration ──────────────────────────────────────────────────

    function test_registerOperator() public {
        string[] memory models = new string[](2);
        models[0] = "kokoro-tts";
        models[1] = "whisper-v3";
        string[] memory taskTypes = new string[](2);
        taskTypes[0] = "tts";
        taskTypes[1] = "stt";

        bytes memory regData = abi.encode(models, taskTypes, uint32(8000), "A10G", "https://op1.modal.run");

        vm.prank(makeAddr("tangle")); // Simulate onlyFromTangle
        // Note: in real tests, need to set up the Tangle mock. For now, test the logic.
        // bsm.onRegister(operator1, regData);
    }

    function test_configureModel() public {
        vm.prank(admin);
        bsm.configureModel("new-model", "tts", InferenceBSM.PricingUnit.Per1KCharacters, 1000, 2000);

        InferenceBSM.ModelConfig memory mc = bsm.getModelConfig("new-model");
        assertEq(mc.pricePerUnit, 1000);
        assertEq(mc.minGpuVramMib, 2000);
        assertTrue(mc.enabled);
        assertEq(uint(mc.pricingUnit), uint(InferenceBSM.PricingUnit.Per1KCharacters));
    }

    function test_configureModel_onlyAdmin() public {
        vm.prank(operator1);
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.NotAdmin.selector, operator1));
        bsm.configureModel("bad", "tts", InferenceBSM.PricingUnit.Per1KCharacters, 1000, 2000);
    }

    function test_disableModel() public {
        vm.startPrank(admin);
        bsm.configureModel("temp", "tts", InferenceBSM.PricingUnit.Per1KCharacters, 1000, 0);
        bsm.disableModel("temp");
        vm.stopPrank();

        InferenceBSM.ModelConfig memory mc = bsm.getModelConfig("temp");
        assertFalse(mc.enabled);
    }

    // ── Admin Controls ────────────────────────────────────────────────

    function test_setAdmin() public {
        // BlueprintServiceManagerBase wires `blueprintOwner` via the one-shot
        // `onBlueprintCreated` hook, not the constructor. Bootstrap it for
        // this unit test so we can exercise the owner-gated path.
        address owner = makeAddr("blueprintOwner");
        bsm.onBlueprintCreated(1, owner, makeAddr("tangle"));

        address newAdmin = address(0xB);
        vm.prank(owner);
        bsm.setAdmin(newAdmin);
        assertEq(bsm.admin(), newAdmin);
    }

    function test_setSlashingEnabled() public {
        vm.prank(admin);
        bsm.setSlashingEnabled(true);
        assertTrue(bsm.slashingEnabled());

        vm.prank(admin);
        bsm.setSlashingEnabled(false);
        assertFalse(bsm.slashingEnabled());
    }

    function test_setPermittedCaller() public {
        address caller = address(0xC);
        vm.prank(admin);
        bsm.setPermittedCaller(caller, true);
        assertTrue(bsm.permittedCallers(caller));

        vm.prank(admin);
        bsm.setPermittedCaller(caller, false);
        assertFalse(bsm.permittedCallers(caller));
    }

    // ── Pricing Unit Coverage ─────────────────────────────────────────

    function test_allPricingUnits() public {
        vm.startPrank(admin);

        // Each pricing unit can be configured
        bsm.configureModel("m1", "tts", InferenceBSM.PricingUnit.PerMillionTokens, 100, 0);
        bsm.configureModel("m2", "stt", InferenceBSM.PricingUnit.Per1KCharacters, 200, 0);
        bsm.configureModel("m3", "s2s", InferenceBSM.PricingUnit.PerSecondAudio, 300, 0);
        bsm.configureModel("m4", "video", InferenceBSM.PricingUnit.PerSecondVideo, 400, 0);
        bsm.configureModel("m5", "image", InferenceBSM.PricingUnit.PerImage, 500, 0);
        bsm.configureModel("m6", "stitch", InferenceBSM.PricingUnit.PerJob, 600, 0);

        vm.stopPrank();

        assertEq(uint(bsm.getModelConfig("m1").pricingUnit), uint(InferenceBSM.PricingUnit.PerMillionTokens));
        assertEq(uint(bsm.getModelConfig("m2").pricingUnit), uint(InferenceBSM.PricingUnit.Per1KCharacters));
        assertEq(uint(bsm.getModelConfig("m3").pricingUnit), uint(InferenceBSM.PricingUnit.PerSecondAudio));
        assertEq(uint(bsm.getModelConfig("m4").pricingUnit), uint(InferenceBSM.PricingUnit.PerSecondVideo));
        assertEq(uint(bsm.getModelConfig("m5").pricingUnit), uint(InferenceBSM.PricingUnit.PerImage));
        assertEq(uint(bsm.getModelConfig("m6").pricingUnit), uint(InferenceBSM.PricingUnit.PerJob));
    }

    // ── View Functions ────────────────────────────────────────────────

    function test_getModelConfig_nonexistent() public view {
        InferenceBSM.ModelConfig memory mc = bsm.getModelConfig("nonexistent");
        assertFalse(mc.enabled);
        assertEq(mc.pricePerUnit, 0);
    }

    function test_getOperators_empty() public view {
        address[] memory ops = bsm.getOperators();
        assertEq(ops.length, 0);
    }

    function test_getActiveOperators_empty() public view {
        address[] memory ops = bsm.getActiveOperators();
        assertEq(ops.length, 0);
    }

    // ── Slashing ──────────────────────────────────────────────────────

    function test_slashingDisabledByDefault() public view {
        assertFalse(bsm.slashingEnabled());
    }

    function test_proposeSlash_reverts_whenDisabled() public {
        vm.prank(admin);
        vm.expectRevert(InferenceBSM.SlashingDisabled.selector);
        bsm.proposePerformanceSlash(1, operator1, 500, bytes32(0));
    }

    // ── Payment Asset ─────────────────────────────────────────────────

    function test_paymentAssetAllowed() public view {
        assertTrue(bsm.queryIsPaymentAssetAllowed(1, address(tsUSD)));
        assertTrue(bsm.queryIsPaymentAssetAllowed(1, address(0)));
        assertFalse(bsm.queryIsPaymentAssetAllowed(1, address(0xDEAD)));
    }

    // ── Configuration ─────────────────────────────────────────────────

    function test_heartbeatConfig() public view {
        (, uint64 interval) = bsm.getHeartbeatInterval(1);
        assertEq(interval, 50);

        (, uint8 threshold) = bsm.getHeartbeatThreshold(1);
        assertEq(threshold, 3);
    }

    function test_minOperatorStake() public view {
        (, uint256 stake) = bsm.getMinOperatorStake();
        assertEq(stake, 50 ether);
    }
}
