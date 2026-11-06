// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {RegisterBlueprint} from "tangle-inference-core/script/RegisterBlueprint.s.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";
contract DeployInference is RegisterBlueprint {
    function _deployManager() internal override returns(address) {
        return _deployManagerFor(vm.envAddress("PAYMENT_TOKEN"),vm.envAddress("TANGLE_CORE"));
    }
    function _deployManagerFor(address token, address registry) internal returns(address) {
        address admin = vm.envOr("ADMIN_ADDRESS",vm.addr(vm.envUint("PRIVATE_KEY")));
        return address(new InferenceBSM(token,admin,registry));
    }
    function _guardNetwork(address registry) internal view override {
        require(registry != 0xff137B9C879c47C28ce389e84501925438AB4cdA, "Tempo: use DeployTempo.s.sol");
    }
}
