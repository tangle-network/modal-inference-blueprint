//! Run the same provider adapter as the chain service, without a chain runner.
use modal_inference::{app_state, config::OperatorConfig, server::build_router};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    dotenvy::dotenv().ok();
    let config = Arc::new(OperatorConfig::load(None)?);
    let address = format!("{}:{}", config.server.host, config.server.port);
    let state = app_state(config)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
