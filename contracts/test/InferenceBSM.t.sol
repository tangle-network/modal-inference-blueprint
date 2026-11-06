// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Test} from "forge-std/Test.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
contract InferenceBSMTest is Test {
    InferenceBSM bsm;
    address operator = address(0x1111);
    address attacker = address(0x9999);
    function setUp() public {
        bsm = new InferenceBSM(address(0xbeef),address(this));
        bsm.onBlueprintCreated(1,address(this),address(this));
        bsm.configureModel("voice","tts",InferenceBSM.PricingUnit.Per1KCharacters,5000,24000);
    }
    function payload(uint32 count,uint32 vram) internal pure returns(bytes memory) {
        string[] memory models=new string[](1); models[0]="voice";
        string[] memory tasks=new string[](1); tasks[0]="tts";
        return abi.encode(models,tasks,count,vram,"nvidia-l4","https://provider.example.com");
    }
    function test_realRegistrationLifecycleUsesPerGpuFacts() public {
        bsm.onRegister(operator,payload(2,24000));
        assertTrue(bsm.canJoin(1,operator));
        (bool ok,bytes memory data)=address(bsm).staticcall(abi.encodeWithSignature("operators(address)",operator));
        assertTrue(ok);
        (uint32 count,uint32 vram)=abi.decode(data,(uint32,uint32));
        assertEq(count,2); assertEq(vram,24000);
        assertEq(bsm.getOperators().length,1);
        bsm.onUnregister(operator);
        assertFalse(bsm.canJoin(1,operator));
        assertEq(bsm.getActiveOperators().length,0);
    }
    function test_moreSmallGpusCannotSatisfyPerGpuVram() public {
        vm.expectRevert(abi.encodeWithSelector(InferenceBSM.InsufficientGpu.selector,uint32(24000),uint32(12000)));
        bsm.onRegister(operator,payload(8,12000));
        vm.expectRevert(); bsm.onRegister(operator,payload(0,24000));
    }
    function test_foreignCallerCannotRegisterOrChangeRequirements() public {
        bytes memory registration=payload(1,24000);
        vm.prank(attacker); vm.expectRevert(); bsm.onRegister(operator,registration);
        vm.prank(attacker); vm.expectRevert();
        bsm.configureModel("voice","tts",InferenceBSM.PricingUnit.Per1KCharacters,0,0);
        assertEq(bsm.getOperators().length,0);
    }
}
