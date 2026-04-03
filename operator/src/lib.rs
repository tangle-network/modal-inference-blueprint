//! modal-inference-blueprint — Tangle Blueprint for serving voice AI models via Modal.
//!
//! Operators deploy Modal apps (any of the 32 voice model scripts),
//! then run this blueprint which provides:
//! - Tangle registration + heartbeat
//! - OpenAI-compatible HTTP proxy
//! - Prometheus metrics + on-chain metric submission
//! - Auto-registration with Tangle marketplace
//! - Billing via x402/ShieldedCredits
//!
//! The blueprint doesn't run models — it proxies to Modal deployments.

pub mod billing;
pub mod config;
pub mod idle;
pub mod metrics;
pub mod proxy;
pub mod qos;
pub mod registry;
pub mod server;

use alloy_sol_types::sol;
use blueprint_sdk::Job;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::std::sync::Arc;

use crate::billing::{BillingClient, NonceStore};
use crate::config::OperatorConfig;
use crate::idle::IdleManager;
use crate::proxy::ModelRegistry;
use crate::server::{AppState, build_router};

// ---------------------------------------------------------------------------
// On-chain ABI types
// ---------------------------------------------------------------------------

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct InferenceRequest {
        string model;
        bytes inputData;
        string inputType;     // "text", "audio"
        string outputType;    // "audio", "text", "json"
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct InferenceResult {
        bytes outputData;
        uint32 unitsConsumed;  // characters for TTS, seconds*100 for STT
        string outputType;
        uint32 latencyMs;
    }
}

pub const INFERENCE_JOB: u8 = 0;

/// Tangle job router — single generic inference job.
/// The model field in the request determines which Modal endpoint to hit.
pub fn router() -> Router {
    Router::new()
        .route(INFERENCE_JOB, run_inference.layer(TangleLayer).layer(blueprint_sdk::tee::TeeLayer::new()))
}

/// Generic inference job handler.
/// Routes to the appropriate Modal endpoint based on the model field.
#[debug_job]
pub async fn run_inference(
    TangleArg(req): TangleArg<InferenceRequest>,
) -> TangleResult<InferenceResult> {
    // The actual proxy happens in the HTTP server.
    // On-chain jobs are for billing verification — the real traffic
    // goes through the HTTP proxy directly.
    //
    // This handler exists so the blueprint is valid on Tangle
    // and can receive on-chain job calls if needed.

    TangleResult(InferenceResult {
        outputData: Vec::new().into(),
        unitsConsumed: 0,
        outputType: req.outputType,
        latencyMs: 0,
    })
}

// ---------------------------------------------------------------------------
// Background service: HTTP server
// ---------------------------------------------------------------------------

pub struct ModalInferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl ModalInferenceServer {
    pub fn new(config: OperatorConfig) -> Self {
        Self { config: Arc::new(config) }
    }
}

impl BackgroundService for ModalInferenceServer {
    async fn start(
        &self,
    ) -> Result<
        tokio::sync::oneshot::Receiver<Result<(), blueprint_sdk::runner::error::RunnerError>>,
        blueprint_sdk::runner::error::RunnerError,
    > {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            let registry = ModelRegistry::new(config.models.clone());

            // Start idle manager if configured
            let idle_mgr = if config.cost.idle_shutdown_minutes > 0 {
                let mgr = IdleManager::new(
                    config.cost.idle_shutdown_minutes,
                    config.cost.idle_check_interval_minutes,
                );
                for m in &config.models {
                    mgr.register_model(&m.name, &m.modal_endpoint).await;
                }
                let checker = mgr.clone();
                tokio::spawn(async move { checker.run_idle_checker().await });
                tracing::info!(
                    idle_mins = config.cost.idle_shutdown_minutes,
                    check_mins = config.cost.idle_check_interval_minutes,
                    "Idle shutdown enabled"
                );
                Some(mgr)
            } else {
                None
            };

            // Initialize billing client when billing is required
            let (billing, operator_address) = if config.billing.required {
                match BillingClient::new(config.clone()).await {
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

            let nonce_store = Arc::new(NonceStore::load(
                config.billing.nonce_store_path.clone(),
            ));

            let state = Arc::new(AppState {
                registry,
                config: (*config).clone(),
                idle_manager: idle_mgr,
                billing,
                nonce_store,
                operator_address,
            });

            // Background health check loop — probes all Modal endpoints periodically
            {
                let state_clone = state.clone();
                let interval_secs = config.qos.metrics_interval_secs.max(30);
                tokio::spawn(async move {
                    let mut interval = tokio::time::interval(
                        std::time::Duration::from_secs(interval_secs),
                    );
                    loop {
                        interval.tick().await;
                        let results = state_clone.registry.health_check_all().await;
                        let healthy = results.iter().filter(|r| r.status == "ok").count();
                        let total = results.len();
                        if healthy < total {
                            tracing::warn!(healthy, total, "Health check: some models unhealthy");
                            for r in &results {
                                if r.status != "ok" {
                                    tracing::warn!(model = %r.name, status = %r.status, "Model unhealthy");
                                }
                            }
                        } else {
                            tracing::debug!(healthy, total, "Health check: all models OK");
                        }
                    }
                });
                tracing::info!(interval_secs, "Background health check loop started");
            }

            let app = build_router(state);
            let addr = format!("{}:{}", config.server.host, config.server.port);

            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Err(blueprint_sdk::runner::error::RunnerError::Other(
                        e.to_string().into(),
                    )));
                    return;
                }
            };

            tracing::info!(addr = %addr, "Modal inference server started");

            if let Err(e) = axum::serve(listener, app).await {
                let _ = tx.send(Err(blueprint_sdk::runner::error::RunnerError::Other(
                    e.to_string().into(),
                )));
            }
        });

        Ok(rx)
    }
}
