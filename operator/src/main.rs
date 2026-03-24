//! Binary entrypoint — BlueprintRunner + QoS wiring.
//!
//! Starts the operator with:
//! 1. Config (Modal endpoints, pricing, QoS)
//! 2. Tangle integration (job router, producer/consumer)
//! 3. QoS service (heartbeat + on-chain metrics submission)
//! 4. HTTP server background service (proxy to Modal)

use std::sync::Arc;

use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use modal_inference::config::OperatorConfig;
use modal_inference::qos::{OperatorMetricsSource, TangleHeartbeatConsumer};
use modal_inference::registry::register_with_gateway;
use modal_inference::ModalInferenceServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();
    dotenvy::dotenv().ok();

    tracing::info!("Modal Inference Blueprint starting...");

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config: {e}")))?;

    tracing::info!(name = %config.name, models = config.models.len(), "Config loaded");

    for model in &config.models {
        tracing::info!(name = %model.name, task = %model.task_type, endpoint = %model.modal_endpoint, "Model configured");
    }

    // Marketplace registration (best-effort)
    if let Err(e) = register_with_gateway(&config).await {
        tracing::warn!(error = %e, "Marketplace registration failed (standalone mode)");
    }

    // Tangle environment
    let env = BlueprintEnvironment::load()?;
    let tangle_client = env.tangle_client().await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env.protocol_settings.tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service_id".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client);

    // ── QoS: heartbeat + on-chain metrics submission ────────────────────
    if config.qos.heartbeat_interval_secs > 0 {
        let metrics_source = Arc::new(OperatorMetricsSource) as Arc<dyn blueprint_qos::heartbeat::MetricsSource>;
        let heartbeat_consumer = Arc::new(TangleHeartbeatConsumer);

        let registry_addr = config.tangle.status_registry_address.parse().unwrap_or_default();

        let heartbeat_ctx = blueprint_qos::HeartbeatContext {
            consumer: heartbeat_consumer,
            http_rpc_endpoint: config.tangle.rpc_url.clone(),
            keystore_uri: config.tangle.operator_key.clone(),
            status_registry_address: registry_addr,
            dry_run: false,
            metrics_source: Some(metrics_source),
        };

        let qos_cfg = blueprint_qos::QoSConfig {
            heartbeat: Some(blueprint_qos::HeartbeatConfig {
                interval_secs: config.qos.heartbeat_interval_secs,
                jitter_percent: 10,
                service_id,
                blueprint_id: config.tangle.blueprint_id,
                max_missed_heartbeats: 5,
                status_registry_address: registry_addr,
            }),
            ..Default::default()
        };

        match blueprint_qos::QoSService::new(qos_cfg, Some(heartbeat_ctx)).await {
            Ok(qos) => {
                tracing::info!(interval = config.qos.heartbeat_interval_secs, "QoS heartbeat started");
                tokio::spawn(async move {
                    if let Err(e) = qos.wait_for_completion().await {
                        tracing::error!(error = %e, "QoS service stopped");
                    }
                });
            }
            Err(e) => {
                tracing::warn!(error = %e, "QoS failed to start (heartbeats disabled)");
            }
        }
    } else {
        tracing::info!("QoS heartbeat disabled (interval = 0)");
    }

    // HTTP server background service
    let server = ModalInferenceServer::new(config);

    tracing::info!("Starting BlueprintRunner...");

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(modal_inference::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(server)
        .run()
        .await?;

    Ok(())
}
