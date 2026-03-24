//! Binary entrypoint — BlueprintRunner wiring only.
//!
//! This is the binary that operators run. It:
//! 1. Loads config (which Modal endpoints to proxy)
//! 2. Loads Tangle environment
//! 3. Registers with Tangle marketplace
//! 4. Starts the BlueprintRunner with:
//!    - Job router (for on-chain inference jobs)
//!    - Tangle producer/consumer (for chain events)
//!    - HTTP server background service (for off-chain traffic)

use std::sync::Arc;

use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use modal_inference::config::OperatorConfig;
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

    // Load operator config
    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config: {e}")))?;

    tracing::info!(
        name = %config.name,
        models = config.models.len(),
        "Operator config loaded"
    );

    for model in &config.models {
        tracing::info!(
            name = %model.name,
            task = %model.task_type,
            endpoint = %model.modal_endpoint,
            "Model endpoint configured"
        );
    }

    // Register with Tangle marketplace (best-effort, non-fatal)
    if let Err(e) = register_with_gateway(&config).await {
        tracing::warn!(error = %e, "Marketplace registration failed (operating standalone)");
    }

    // Load Tangle environment
    let env = BlueprintEnvironment::load()?;

    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service_id configured".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client);

    // Background service: HTTP server that proxies to Modal
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
