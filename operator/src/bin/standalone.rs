//! Standalone HTTP server — runs the operator without Tangle SDK.
//! For local testing and development.
//!
//! Usage: cargo run --bin standalone

use modal_inference::config::OperatorConfig;
use modal_inference::proxy::ModelRegistry;
use modal_inference::server::{build_router, AppState};
use std::sync::Arc;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_log();
    dotenvy::dotenv().ok();

    tracing::info!("Modal Inference Operator (standalone mode)");

    let config = OperatorConfig::load(None)?;

    tracing::info!(
        name = %config.name,
        models = config.models.len(),
        port = config.server.port,
        "Config loaded"
    );

    for model in &config.models {
        tracing::info!(
            name = %model.name,
            task = %model.task_type,
            endpoint = %model.modal_endpoint,
            "Model"
        );
    }

    let registry = ModelRegistry::new(config.models.clone());
    let state = Arc::new(AppState {
        registry,
        config: config.clone(),
    });

    let app = build_router(state);
    let addr = format!("{}:{}", config.server.host, config.server.port);

    tracing::info!(%addr, "HTTP server starting");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
