//! QoS integration — bridges our Prometheus metrics to the Tangle on-chain
//! heartbeat + metrics submission system.
//!
//! Uses the blueprint-qos crate's `MetricsSource` trait to feed our
//! operator metrics into the heartbeat loop, which submits them on-chain
//! via `submitHeartbeat` on the IOperatorStatusRegistry contract.

use crate::metrics;
use blueprint_qos::heartbeat::MetricsSource;
use std::future::Future;
use std::pin::Pin;

/// Bridges our Prometheus/atomic metrics to the QoS on-chain submission.
/// Implements `MetricsSource` so the heartbeat loop can read our metrics
/// and include them in each heartbeat transaction.
pub struct OperatorMetricsSource;

impl MetricsSource for OperatorMetricsSource {
    /// Read all pending on-chain metrics.
    /// Called by the heartbeat loop before each submission.
    fn get_custom_metrics(&self) -> Pin<Box<dyn Future<Output = Vec<(String, u64)>> + Send + '_>> {
        Box::pin(async {
            metrics::on_chain_metrics()
        })
    }

    /// Clear metrics after successful on-chain submission.
    /// We don't actually clear our atomics — they're cumulative counters.
    /// The chain can diff between submissions to compute per-interval rates.
    fn clear_custom_metrics(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {
            // Cumulative counters — no clearing needed.
            // The BSM contract diffs current vs previous submission.
        })
    }
}
