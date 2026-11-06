// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {DeployInference} from "./Deploy.s.sol";
import {console2} from "forge-std/Script.sol";
import {Types} from "tnt-core/libraries/Types.sol";
import {IInferenceBlueprintRegistry} from "tangle-inference-core/script/RegisterBlueprint.s.sol";
interface IERC20Decimals {
    function decimals() external view returns (uint8);
}
interface ITempoSettlement {
    function setBlueprintSettlementAsset(uint64 blueprintId, address asset) external;
    function setJobEventRates(uint64 blueprintId, uint8[] calldata jobs, uint256[] calldata rates) external;
    function getBlueprintSettlementAsset(uint64 blueprintId) external view returns (address);
    function getJobEventRate(uint64 blueprintId, uint8 jobIndex) external view returns (uint256);
}
contract DeployTempo is DeployInference {
    address constant TANGLE = 0xff137B9C879c47C28ce389e84501925438AB4cdA;
    address constant PATH_USD = 0x20C0000000000000000000000000000000000000;
    function run() external override {
        address registry = vm.envOr("TANGLE_CORE", TANGLE);
        address token = vm.envOr("PAYMENT_TOKEN", PATH_USD);
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        uint8 decimals = IERC20Decimals(token).decimals();
        require(decimals <= 18, "unsupported settlement decimals");
        uint256 floor = vm.envOr("EVENT_RATE", _defaultFloor(decimals));
        require(floor > 0 && floor <= 1_000_000 * (10 ** uint256(decimals)), "EVENT_RATE out of range for settlement decimals");
        vm.startBroadcast(vm.envUint("PRIVATE_KEY"));
        address manager = _deployManagerFor(token, registry);
        Types.BlueprintDefinition memory definition = _buildDefinition(manager);
        definition.metadataUri = "tangle:modal-inference";
        definition.config.eventRate = floor;
        uint64 id = IInferenceBlueprintRegistry(registry).createBlueprint(definition);
        ITempoSettlement tangle = ITempoSettlement(registry);
        tangle.setBlueprintSettlementAsset(id, token);
        uint8[] memory jobs = new uint8[](1);
        uint256[] memory rates = new uint256[](1);
        rates[0] = floor;
        tangle.setJobEventRates(id, jobs, rates);
        vm.stopBroadcast();
        require(tangle.getBlueprintSettlementAsset(id) == token, "settlement asset was not pinned");
        require(tangle.getJobEventRate(id, 0) == floor, "inference floor was not written");
        _report(manager, id);
        console2.log("floor (token units):", floor);
    }
    function _defaultFloor(uint8 decimals) internal pure returns (uint256) {
        uint256 floor = (10 ** uint256(decimals)) / 1000;
        return floor == 0 ? 1 : floor;
    }
}
