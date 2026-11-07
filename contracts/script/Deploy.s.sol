// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {InferenceBSM} from "../src/InferenceBSM.sol";

contract DeployInference is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        address token = vm.envAddress("PAYMENT_TOKEN");
        address admin = vm.envAddress("ADMIN_ADDRESS");
        vm.startBroadcast();
        address manager = address(new InferenceBSM(token, admin, registry));
        vm.stopBroadcast();
        console2.log("DEPLOY_INFERENCE_BSM=%s", vm.toString(manager));
    }
}
