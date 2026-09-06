//! Standalone operator — runs HTTP server + metrics reporting without Tangle SDK.
//! For local testing against a real Modal deployment or local endpoints.
//!
//! Usage: cargo run --bin standalone

use blueprint_sdk::std::sync::Arc;
use modal_inference::config::OperatorConfig;
use modal_inference::idle::IdleManager;
use modal_inference::proxy::ModelRegistry;
use modal_inference::server::{build_router, ModalBackend};
use modal_inference::{AppStateBuilder, BillingClient, NonceStore};

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
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
        models = config.modal.models.len(),
        port = config.server.port,
        "Config loaded"
    );

    for model in &config.modal.models {
        tracing::info!(
            name = %model.name,
            task = %model.task_type,
            endpoint = %model.modal_endpoint,
            "Model"
        );
    }

    let config = Arc::new(config);

    // Initialize billing client
    let billing_client = match BillingClient::new(&config.tangle, &config.billing) {
        Ok(client) => {
            tracing::info!(operator = %client.operator_address(), "Billing enabled");
            Arc::new(client)
        }
        Err(e) => {
            tracing::warn!(error = %e, "BillingClient init failed — billing disabled");
            return Err(anyhow::anyhow!(
                "BillingClient required for standalone mode: {e}"
            ));
        }
    };

    let operator_address = billing_client.operator_address();
    let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));

    let registry = ModelRegistry::new(config.modal.models.clone());

    // Idle manager
    let idle_mgr = if config.modal.idle_shutdown_minutes > 0 {
        let mgr = IdleManager::new(
            config.modal.idle_shutdown_minutes,
            config.modal.idle_check_interval_minutes,
        );
        for m in &config.modal.models {
            mgr.register_model(&m.name, &m.modal_endpoint).await;
        }
        let checker = mgr.clone();
        tokio::spawn(async move { checker.run_idle_checker().await });
        tracing::info!(
            idle_mins = config.modal.idle_shutdown_minutes,
            "Idle shutdown enabled"
        );
        Some(mgr)
    } else {
        None
    };

    let backend = ModalBackend::new(config.clone(), registry, idle_mgr);

    let state = AppStateBuilder::new()
        .billing(billing_client)
        .nonce_store(nonce_store)
        .server_config(Arc::new(config.server.clone()))
        .billing_config(Arc::new(config.billing.clone()))
        .operator_address(operator_address)
        .max_concurrent(config.server.max_concurrent_requests)
        .backend(backend)
        .build()
        .map_err(|e| anyhow::anyhow!("AppState build failed: {e}"))?;

    let app = build_router(state);
    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!(%addr, "HTTP server starting");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
