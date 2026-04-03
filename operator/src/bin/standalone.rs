//! Standalone operator — runs HTTP server + metrics reporting without Tangle SDK.
//! For local testing against a real Modal deployment or local endpoints.
//!
//! Usage: cargo run --bin standalone

use modal_inference::billing::{BillingClient, NonceStore};
use modal_inference::config::OperatorConfig;
use modal_inference::idle::IdleManager;
use modal_inference::metrics;
use modal_inference::proxy::ModelRegistry;
use modal_inference::server::{build_router, AppState};
use blueprint_sdk::std::sync::Arc;

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
    tracing::info!(name = %config.name, models = config.models.len(), port = config.server.port, "Config loaded");

    for model in &config.models {
        tracing::info!(name = %model.name, task = %model.task_type, endpoint = %model.modal_endpoint, "Model");
    }

    let registry = ModelRegistry::new(config.models.clone());

    // Idle manager
    let idle_mgr = if config.cost.idle_shutdown_minutes > 0 {
        let mgr = IdleManager::new(config.cost.idle_shutdown_minutes, config.cost.idle_check_interval_minutes);
        for m in &config.models {
            mgr.register_model(&m.name, &m.modal_endpoint).await;
        }
        let checker = mgr.clone();
        tokio::spawn(async move { checker.run_idle_checker().await });
        tracing::info!(idle_mins = config.cost.idle_shutdown_minutes, "Idle shutdown enabled");
        Some(mgr)
    } else {
        None
    };

    // Initialize billing client when billing.required is set
    let (billing, operator_address) = if config.billing.required {
        match BillingClient::new(Arc::new(config.clone())).await {
            Ok(client) => {
                let addr = client.operator_address();
                tracing::info!(operator = %addr, "Billing enabled");
                (Some(Arc::new(client)), Some(addr))
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to initialize BillingClient — billing disabled");
                (None, None)
            }
        }
    } else {
        tracing::info!("Billing disabled (billing.required = false)");
        (None, None)
    };

    let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));

    let state = Arc::new(AppState {
        registry,
        config: config.clone(),
        idle_manager: idle_mgr,
        billing,
        nonce_store,
        operator_address,
    });

    // Metrics reporting loop (logs metrics periodically, simulates on-chain submission)
    let report_interval = config.qos.metrics_interval_secs;
    if report_interval > 0 {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(report_interval));
            loop {
                interval.tick().await;
                let m = metrics::on_chain_metrics();
                let summary: Vec<String> = m.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
                tracing::info!(metrics = %summary.join(", "), "Metrics report");
            }
        });
        tracing::info!(interval_secs = report_interval, "Metrics reporting started");
    }

    let app = build_router(state);
    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!(%addr, "HTTP server starting");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
