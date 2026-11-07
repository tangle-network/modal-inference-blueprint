// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";

interface IERC20Decimals {
    function decimals() external view returns (uint8);
}

interface ITempoSettlement {
    function setBlueprintSettlementAsset(uint64 blueprintId, address asset) external;
    function setJobEventRates(uint64 blueprintId, uint8[] calldata jobs, uint256[] calldata rates) external;
    function getBlueprintSettlementAsset(uint64 blueprintId) external view returns (address);
    function getJobEventRate(uint64 blueprintId, uint8 jobIndex) external view returns (uint256);
}

contract DeployTempo is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        address token = vm.envAddress("PAYMENT_TOKEN");
        uint256 rawId = vm.envUint("BLUEPRINT_ID");
        require(rawId < type(uint64).max, "invalid blueprint id");
        uint64 id = uint64(rawId);
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        uint8 decimals = IERC20Decimals(token).decimals();
        require(decimals <= 18, "unsupported settlement decimals");
        uint256 floor = vm.envOr("EVENT_RATE", _defaultFloor(decimals));
        require(
            floor > 0 && floor <= 1_000_000 * (10 ** uint256(decimals)),
            "EVENT_RATE out of range for settlement decimals"
        );
        ITempoSettlement tangle = ITempoSettlement(registry);
        uint8[] memory jobs = new uint8[](1);
        uint256[] memory rates = new uint256[](1);
        rates[0] = floor;
        vm.startBroadcast();
        tangle.setBlueprintSettlementAsset(id, token);
        tangle.setJobEventRates(id, jobs, rates);
        vm.stopBroadcast();
        require(tangle.getBlueprintSettlementAsset(id) == token, "settlement asset was not pinned");
        require(tangle.getJobEventRate(id, 0) == floor, "inference floor was not written");
        console2.log("floor (token units):", floor);
    }

    function _defaultFloor(uint8 decimals) internal pure returns (uint256) {
        uint256 floor = (10 ** uint256(decimals)) / 1000;
        return floor == 0 ? 1 : floor;
    }
}
