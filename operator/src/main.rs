//! Binary entrypoint — BlueprintRunner + Tangle integration.
//!
//! Starts the operator with:
//! 1. Config (Modal endpoints, pricing)
//! 2. Tangle integration (job router, producer/consumer)
//! 3. HTTP server background service (proxy to Modal)

use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use modal_inference::config::OperatorConfig;
use modal_inference::ModalInferenceServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
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

    tracing::info!(
        name = %config.name,
        models = config.modal.models.len(),
        "Config loaded"
    );

    for model in &config.modal.models {
        tracing::info!(
            name = %model.name,
            task = %model.task_type,
            endpoint = %model.modal_endpoint,
            "Model configured"
        );
    }

    // Tangle environment
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
        .ok_or_else(|| blueprint_sdk::Error::Other("No service_id".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client);

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
