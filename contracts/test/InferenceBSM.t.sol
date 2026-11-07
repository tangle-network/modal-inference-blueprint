// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Test} from "forge-std/Test.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
contract InferenceBSMTest is Test {
    InferenceBSM bsm;
    address operator = address(0x1111);
    address attacker = address(0x9999);
    function setUp() public {
        bsm = new InferenceBSM(address(0xbeef),address(this),address(this));
        bsm.onBlueprintCreated(1,address(this),address(this));
        bsm.configureModel("voice","tts",InferenceBSM.PricingUnit.Per1KCharacters,5000,24000);
    }
    function payload(uint32 vram) internal pure returns(bytes memory) {
        string[] memory models=new string[](1); models[0]="voice";
        string[] memory tasks=new string[](1); tasks[0]="tts";
        return abi.encode(models,tasks,vram,"nvidia-l4","https://provider.example.com");
    }
    function test_liveRegistrationWireAndLifecycle() public {
        bsm.onRegister(operator,payload(24000));
        assertTrue(bsm.canJoin(1,operator));
        (bool ok,bytes memory data)=address(bsm).staticcall(abi.encodeWithSignature("operators(address)",operator));
        assertTrue(ok);
        uint32 vram=abi.decode(data,(uint32));
        assertEq(vram,24000);
        assertEq(bsm.getOperators().length,1);
        bsm.onUnregister(operator);
        assertFalse(bsm.canJoin(1,operator));
        assertEq(bsm.getActiveOperators().length,0);
    }
    function test_insufficientAdvertisedVramRejected() public {
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.InsufficientGpu.selector,uint32(24000),uint32(12000)));
        bsm.onRegister(operator,payload(12000));
    }
    function test_foreignCallerCannotRegisterOrChangeRequirements() public {
        bytes memory registration=payload(24000);
        vm.prank(attacker); vm.expectRevert(); bsm.onRegister(operator,registration);
        vm.prank(attacker); vm.expectRevert();
        bsm.configureModel("voice","tts",InferenceBSM.PricingUnit.Per1KCharacters,0,0);
        assertEq(bsm.getOperators().length,0);
    }
    function test_operatorsCannotRateThemselvesOrEachOther() public {
        address other = address(0x2222);
        bsm.onRegister(operator,payload(24000)); bsm.onRegister(other,payload(24000));
        vm.prank(operator); vm.expectRevert(); bsm.submitMetrics(other,0,0,1000,1000);
        vm.prank(operator); vm.expectRevert(); bsm.submitMetrics(operator,10000,0,1000,0);
        address collector = address(0xc011);
        bsm.setPermittedCaller(collector,true);
        vm.prank(collector); bsm.submitMetrics(other,9999,1,1000,0);
    }
    function test_onlyAutoSuspensionIsSelfLiftable() public {
        address collector = address(0xc011);
        bsm.setPermittedCaller(collector,true); bsm.setSlashingEnabled(true);
        bsm.onRegister(operator,payload(24000));
        bsm.setOperatorSuspended(operator,true,"admin");
        vm.prank(collector); bsm.submitMetrics(operator,10000,1,1000,0);
        vm.prank(operator); vm.expectRevert(); bsm.reactivate();
        bsm.setOperatorSuspended(operator,false,"admin");
        vm.prank(collector); bsm.submitMetrics(operator,5000,1,1000,500);
        vm.prank(operator); vm.expectRevert(); bsm.reactivate();
        vm.prank(collector); bsm.submitMetrics(operator,10000,1,1000,0);
        vm.prank(operator); bsm.reactivate();
    }
    function test_foreignFirstBindCannotClaimManager() public {
        InferenceBSM fresh = new InferenceBSM(address(0xbeef),address(this),address(this));
        vm.prank(attacker); vm.expectRevert(); fresh.onBlueprintCreated(999,attacker,attacker);
        fresh.onBlueprintCreated(1,address(this),address(this));
        assertEq(fresh.blueprintOwner(),address(this));
        vm.expectRevert(); fresh.onBlueprintCreated(2,attacker,address(this));
    }

}
