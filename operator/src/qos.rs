//! QoS integration — bridges our metrics to the Tangle on-chain heartbeat system.
//!
//! Implements:
//! - `MetricsSource`: feeds 13 on-chain metrics into each heartbeat
//! - `TangleHeartbeatConsumer`: sends heartbeat on-chain
//!
//! The HeartbeatService calls get_custom_metrics() on each tick, includes them
//! in the HeartbeatStatus, signs it, and calls submitHeartbeat() on-chain
//! via the IOperatorStatusRegistry contract.

use crate::metrics;
use blueprint_qos::heartbeat::{HeartbeatConsumer, HeartbeatStatus, MetricsSource};
use blueprint_qos::error::Result;
use blueprint_std::future::Future;
use blueprint_std::pin::Pin;

/// Bridges our Prometheus/atomic metrics to the QoS on-chain submission.
pub struct OperatorMetricsSource;

impl MetricsSource for OperatorMetricsSource {
    fn get_custom_metrics(&self) -> Pin<Box<dyn Future<Output = Vec<(String, u64)>> + Send + '_>> {
        Box::pin(async { metrics::on_chain_metrics() })
    }

    fn clear_custom_metrics(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {
            // Cumulative counters — BSM contract diffs between submissions.
        })
    }
}

/// On-chain heartbeat consumer. In practice, the HeartbeatService from
/// blueprint-qos handles signing + tx submission internally using the
/// keystore_uri + http_rpc_endpoint. This consumer logs + increments counters.
pub struct TangleHeartbeatConsumer;

impl HeartbeatConsumer for TangleHeartbeatConsumer {
    fn send_heartbeat(
        &self,
        status: &HeartbeatStatus,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>> {
        let service_id = status.service_id;
        let blueprint_id = status.blueprint_id;
        let block = status.block_number;
        Box::pin(async move {
            tracing::info!(service_id, blueprint_id, block, "Heartbeat submitted");
            metrics::HEARTBEATS_SENT.inc();
            Ok(())
        })
    }
}
