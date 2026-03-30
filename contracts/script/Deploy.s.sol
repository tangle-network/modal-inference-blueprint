// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "forge-std/Script.sol";
import "../src/InferenceBSM.sol";

contract DeployInferenceBSM is Script {
    function run() external {
        address tsUSD = vm.envAddress("TSUSD_ADDRESS");
        address admin = vm.envAddress("ADMIN_ADDRESS");

        vm.startBroadcast();
        InferenceBSM bsm = new InferenceBSM(tsUSD, admin);
        vm.stopBroadcast();

        console.log("InferenceBSM deployed at:", address(bsm));
        console.log("  tsUSD:", tsUSD);
        console.log("  admin:", admin);
    }
}
