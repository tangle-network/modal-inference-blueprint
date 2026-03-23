// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";
import { SlashingLib } from "tnt-core/libraries/SlashingLib.sol";
import { EnumerableSet } from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

/// @title VoiceInferenceBSM
/// @notice Blueprint Service Manager for voice AI inference on the ph0ny network.
///
/// Slashing architecture:
///   - Heartbeat-based slashing: handled by Tangle Core protocol automatically.
///     We just configure intervals/thresholds, protocol calls onUnappliedSlash/onSlash.
///   - Performance-based slashing: BSM-initiated via SlashingLib for custom violations
///     (poor uptime, high error rate) with dispute windows. Admin-toggleable.
///
/// The protocol handles:
///   - Heartbeat monitoring + missed heartbeat detection
///   - Calling onUnappliedSlash when threshold is breached
///   - Executing the actual stake deduction via onSlash
///   - Exit queue enforcement
///
/// This contract handles:
///   - Operator registration with GPU/model validation
///   - On-chain metrics tracking (uptime, latency, errors)
///   - Custom slash proposals for poor performance (via SlashingLib)
///   - Admin controls (toggle slashing, manage operators, configure models)
contract VoiceInferenceBSM is BlueprintServiceManagerBase {
    using EnumerableSet for EnumerableSet.AddressSet;

    // ═══════════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════════

    error OperatorNotRegistered(address operator);
    error ModelNotEnabled(string model);
    error InsufficientGpu(uint32 required, uint32 provided);
    error SlashingDisabled();
    error NotAdmin(address caller);
    error NotPermittedCaller(address caller);
    error UptimeBelowThreshold(uint256 current, uint256 required);

    // ═══════════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    event OperatorRegistered(address indexed operator, string[] models, uint32 gpuVramMib, string endpoint);
    event OperatorStatusChanged(address indexed operator, bool suspended, string reason);
    event MetricsSubmitted(address indexed operator, uint256 uptimeBps, uint256 avgLatencyMs, uint64 requestsTotal);
    event ModelConfigured(string model, string taskType, uint64 pricePer1KUnits, uint32 minGpuVramMib);
    event SlashingToggled(bool enabled);
    event AdminSet(address indexed admin);
    event PermittedCallerSet(address indexed caller, bool permitted);

    // ═══════════════════════════════════════════════════════════════════════════
    // STRUCTS
    // ═══════════════════════════════════════════════════════════════════════════

    struct OperatorInfo {
        string[] models;
        string[] taskTypes;
        uint32 gpuVramMib;
        string gpuModel;
        string endpoint;
        bool active;
        bool suspended;
        uint64 registeredAt;
        uint256 uptimeBps;        // 0-10000 basis points
        uint256 avgLatencyMs;
        uint64 requestsTotal;
        uint64 requestsError;
        uint64 lastMetricsBlock;
    }

    struct ModelConfig {
        string taskType;
        uint64 pricePer1KUnits;
        uint32 minGpuVramMib;
        bool enabled;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTANTS
    // ═══════════════════════════════════════════════════════════════════════════

    uint256 public constant MIN_OPERATOR_STAKE = 50 ether;
    uint256 public constant MIN_UPTIME_BPS = 9000;      // 90%
    uint16 public constant PERFORMANCE_SLASH_BPS = 500;  // 5%

    // ═══════════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════════

    address public immutable tsUSD;
    address public admin;
    bool public slashingEnabled;

    // SlashingLib storage — for BSM-initiated performance slashing
    SlashingLib.SlashState private _slashState;
    mapping(uint64 => SlashingLib.SlashProposal) private _slashProposals;

    mapping(address => OperatorInfo) public operators;
    mapping(bytes32 => ModelConfig) public modelConfigs;
    EnumerableSet.AddressSet private _operatorSet;
    mapping(address => bool) public permittedCallers;

    // ═══════════════════════════════════════════════════════════════════════════
    // MODIFIERS
    // ═══════════════════════════════════════════════════════════════════════════

    modifier onlyAdmin() {
        if (msg.sender != admin && msg.sender != blueprintOwner) revert NotAdmin(msg.sender);
        _;
    }

    modifier onlyPermittedOrOperator(address operator) {
        if (msg.sender != operator && !permittedCallers[msg.sender] && msg.sender != admin) {
            revert NotPermittedCaller(msg.sender);
        }
        _;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════

    constructor(address _tsUSD, address _admin) {
        tsUSD = _tsUSD;
        admin = _admin;
        slashingEnabled = false; // OFF by default — turn on when network is mature

        // Initialize SlashingLib with 1-day dispute window
        SlashingLib.initializeConfig(_slashState);
        // Override to 1 day (default is 7 days)
        SlashingLib.updateConfig(_slashState, 1 days, false, 1000); // max 10% slash
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADMIN
    // ═══════════════════════════════════════════════════════════════════════════

    function setSlashingEnabled(bool enabled) external onlyAdmin {
        slashingEnabled = enabled;
        emit SlashingToggled(enabled);
    }

    function setAdmin(address newAdmin) external onlyBlueprintOwner {
        admin = newAdmin;
        emit AdminSet(newAdmin);
    }

    function setPermittedCaller(address caller, bool permitted) external onlyAdmin {
        permittedCallers[caller] = permitted;
        emit PermittedCallerSet(caller, permitted);
    }

    function configureModel(
        string calldata model, string calldata taskType,
        uint64 pricePer1KUnits, uint32 minGpuVramMib
    ) external onlyAdmin {
        bytes32 key = keccak256(bytes(model));
        modelConfigs[key] = ModelConfig(taskType, pricePer1KUnits, minGpuVramMib, true);
        emit ModelConfigured(model, taskType, pricePer1KUnits, minGpuVramMib);
    }

    function disableModel(string calldata model) external onlyAdmin {
        modelConfigs[keccak256(bytes(model))].enabled = false;
    }

    function setOperatorSuspended(address operator, bool suspended, string calldata reason) external onlyAdmin {
        if (!operators[operator].active) revert OperatorNotRegistered(operator);
        operators[operator].suspended = suspended;
        emit OperatorStatusChanged(operator, suspended, reason);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // METRICS
    // ═══════════════════════════════════════════════════════════════════════════

    function submitMetrics(
        address operator, uint256 uptimeBps, uint256 avgLatencyMs,
        uint64 requestsTotal, uint64 requestsError
    ) external onlyPermittedOrOperator(operator) {
        OperatorInfo storage op = operators[operator];
        if (!op.active) revert OperatorNotRegistered(operator);

        op.uptimeBps = uptimeBps;
        op.avgLatencyMs = avgLatencyMs;
        op.requestsTotal = requestsTotal;
        op.requestsError = requestsError;
        op.lastMetricsBlock = uint64(block.number);

        emit MetricsSubmitted(operator, uptimeBps, avgLatencyMs, requestsTotal);

        // Auto-suspend only when slashing is enabled
        if (slashingEnabled && uptimeBps < MIN_UPTIME_BPS && requestsTotal > 100 && !op.suspended) {
            op.suspended = true;
            emit OperatorStatusChanged(operator, true, "Auto-suspended: uptime below 90%");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BSM-INITIATED SLASHING (via SlashingLib — for custom performance violations)
    // Protocol handles heartbeat slashing automatically via onUnappliedSlash/onSlash.
    // This is for ADDITIONAL slashing beyond heartbeats.
    // ═══════════════════════════════════════════════════════════════════════════

    /// @notice Propose slashing an operator for poor performance.
    ///         Goes through SlashingLib dispute window (1 day default).
    function proposePerformanceSlash(
        uint64 serviceId, address operator, uint16 exposureBps, bytes32 evidence
    ) external onlyAdmin returns (uint64 slashId) {
        if (!slashingEnabled) revert SlashingDisabled();
        if (!operators[operator].active) revert OperatorNotRegistered(operator);

        slashId = SlashingLib.proposeSlash(
            _slashState, _slashProposals,
            serviceId, operator, msg.sender,
            PERFORMANCE_SLASH_BPS, exposureBps, evidence,
            false // not instant — goes through dispute window
        );

        operators[operator].suspended = true;
        emit OperatorStatusChanged(operator, true, "Performance slash proposed");
    }

    /// @notice Execute a performance slash after dispute window passes.
    function executePerformanceSlash(uint64 slashId) external onlyAdmin {
        if (!slashingEnabled) revert SlashingDisabled();
        SlashingLib.markExecuted(_slashProposals, slashId, 0);
        // Tangle Core handles actual stake deduction when it sees the execution
    }

    /// @notice Operator disputes a pending slash.
    function disputeSlash(uint64 slashId, string calldata reason) external {
        SlashingLib.disputeSlash(_slashProposals, slashId, msg.sender, reason);
    }

    /// @notice Admin cancels a slash (e.g., dispute was valid).
    function cancelSlash(uint64 slashId, string calldata reason) external onlyAdmin {
        SlashingLib.cancelSlash(_slashProposals, slashId, msg.sender, reason);
        // Unsuspend the operator
        address operator = _slashProposals[slashId].operator;
        if (operators[operator].active) {
            operators[operator].suspended = false;
            emit OperatorStatusChanged(operator, false, "Slash cancelled, reactivated");
        }
    }

    /// @notice Operator reactivates after improving metrics.
    function reactivate() external {
        OperatorInfo storage op = operators[msg.sender];
        if (!op.active) revert OperatorNotRegistered(msg.sender);
        if (op.uptimeBps < MIN_UPTIME_BPS) revert UptimeBelowThreshold(op.uptimeBps, MIN_UPTIME_BPS);
        op.suspended = false;
        emit OperatorStatusChanged(msg.sender, false, "Self-reactivated");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BSM HOOKS — Protocol calls these. We override only what we need.
    // BlueprintServiceManagerBase provides safe defaults for everything else.
    // ═══════════════════════════════════════════════════════════════════════════

    function onRegister(address operator, bytes calldata registrationInputs)
        external payable override onlyFromTangle
    {
        (
            string[] memory models, string[] memory taskTypes,
            uint32 gpuVramMib, string memory gpuModel, string memory endpoint
        ) = abi.decode(registrationInputs, (string[], string[], uint32, string, string));

        for (uint i = 0; i < models.length; i++) {
            bytes32 key = keccak256(bytes(models[i]));
            ModelConfig storage mc = modelConfigs[key];
            if (!mc.enabled) revert ModelNotEnabled(models[i]);
            if (gpuVramMib < mc.minGpuVramMib) revert InsufficientGpu(mc.minGpuVramMib, gpuVramMib);
        }

        operators[operator] = OperatorInfo({
            models: models, taskTypes: taskTypes,
            gpuVramMib: gpuVramMib, gpuModel: gpuModel, endpoint: endpoint,
            active: true, suspended: false,
            registeredAt: uint64(block.timestamp),
            uptimeBps: 10000, avgLatencyMs: 0,
            requestsTotal: 0, requestsError: 0,
            lastMetricsBlock: uint64(block.number)
        });

        _operatorSet.add(operator);
        permittedCallers[operator] = true; // Operator can submit own metrics

        emit OperatorRegistered(operator, models, gpuVramMib, endpoint);
    }

    function onUnregister(address operator) external override onlyFromTangle {
        operators[operator].active = false;
        _operatorSet.remove(operator);
        permittedCallers[operator] = false;
    }

    function onUpdatePreferences(address operator, bytes calldata newPreferences)
        external payable override onlyFromTangle
    {
        operators[operator].endpoint = abi.decode(newPreferences, (string));
    }

    /// @notice Protocol calls this when heartbeat threshold is breached.
    ///         We suspend the operator. Protocol handles the actual slashing.
    function onUnappliedSlash(uint64, bytes calldata offender, uint8)
        external override onlyFromTangle
    {
        address operator = abi.decode(offender, (address));
        if (operators[operator].active) {
            operators[operator].suspended = true;
            emit OperatorStatusChanged(operator, true, "Heartbeat slashing (protocol)");
        }
    }

    /// @notice Protocol calls this when slash is executed (stake deducted).
    function onSlash(uint64, bytes calldata offender, uint8)
        external override onlyFromTangle
    {
        address operator = abi.decode(offender, (address));
        operators[operator].suspended = true;
        emit OperatorStatusChanged(operator, true, "Slashed (protocol)");
    }

    function onJobResult(uint64, uint8, uint64, address operator, bytes calldata, bytes calldata)
        external payable override onlyFromTangle
    {
        if (operators[operator].active) operators[operator].requestsTotal++;
    }

    function canJoin(uint64, address operator) external view override returns (bool) {
        return operators[operator].active && !operators[operator].suspended;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CONFIGURATION — Tell the protocol our preferences
    // ═══════════════════════════════════════════════════════════════════════════

    function getHeartbeatInterval(uint64) external pure override returns (bool, uint64) {
        return (false, 50); // ~5 min at 6s blocks
    }

    function getHeartbeatThreshold(uint64) external pure override returns (bool, uint8) {
        return (false, 3); // 3 missed = protocol slashes
    }

    function getSlashingWindow(uint64) external pure override returns (bool, uint64) {
        return (false, 600); // ~1 hour
    }

    function getExitConfig(uint64) external pure override returns (bool, uint64, uint64, bool) {
        return (false, 86400, 3600, true);
    }

    function getMinOperatorStake() external pure override returns (bool, uint256) {
        return (false, MIN_OPERATOR_STAKE);
    }

    function querySlashingOrigin(uint64) external view override returns (address) {
        return address(this); // This BSM can initiate slashing
    }

    function queryDisputeOrigin(uint64) external view override returns (address) {
        return admin; // Admin handles disputes
    }

    function queryDeveloperPaymentAddress(uint64) external view override returns (address payable) {
        return payable(blueprintOwner);
    }

    function queryIsPaymentAssetAllowed(uint64, address asset) external view override returns (bool) {
        return asset == tsUSD || asset == address(0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW
    // ═══════════════════════════════════════════════════════════════════════════

    function getOperators() external view returns (address[] memory) {
        return _operatorSet.values();
    }

    function getActiveOperators() external view returns (address[] memory) {
        address[] memory all = _operatorSet.values();
        uint256 count;
        for (uint i = 0; i < all.length; i++) {
            if (operators[all[i]].active && !operators[all[i]].suspended) count++;
        }
        address[] memory active = new address[](count);
        uint j;
        for (uint i = 0; i < all.length; i++) {
            if (operators[all[i]].active && !operators[all[i]].suspended) active[j++] = all[i];
        }
        return active;
    }

    function getOperatorReputation(address operator) external view returns (
        uint256 uptimeBps, uint256 avgLatencyMs, uint64 requestsTotal,
        uint64 requestsError, bool suspended, uint64 lastMetricsBlock
    ) {
        OperatorInfo storage op = operators[operator];
        return (op.uptimeBps, op.avgLatencyMs, op.requestsTotal, op.requestsError, op.suspended, op.lastMetricsBlock);
    }

    function getSlashProposal(uint64 slashId) external view returns (SlashingLib.SlashProposal memory) {
        return _slashProposals[slashId];
    }

    receive() external payable {}
}
