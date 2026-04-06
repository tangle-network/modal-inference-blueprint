//! modal-inference-blueprint — Tangle Blueprint for serving multi-modal AI via Modal.
//!
//! Operators deploy Modal apps (any of the 100+ voice/video/image/LLM model
//! scripts), then run this blueprint which provides:
//! - Tangle registration + heartbeat
//! - OpenAI-compatible HTTP proxy to Modal
//! - Prometheus metrics + on-chain metric submission
//! - Billing via x402 / ShieldedCredits with per-task-type pricing
//!
//! All shared operator infrastructure (billing, metrics, health, nonce store,
//! spend-auth validation, x402 payment headers, AppState builder) lives in
//! `tangle-inference-core`. This crate only contains the modal-specific
//! backend (HTTP proxy, model registry, idle manager, task-aware cost model).

pub mod config;
pub mod idle;
pub mod proxy;
pub mod qos;
pub mod server;

// Re-export shared infrastructure so downstream crates can
// `use modal_inference::*`.
pub use tangle_inference_core::{
    billing, metrics, AppState, AppStateBuilder, BillingClient, CostModel, CostParams,
    FlatRequestCostModel, NonceStore, PerCharCostModel, PerImageCostModel, PerSecondCostModel,
    PerTokenCostModel, RequestGuard, SpendAuthPayload, TaskTypeCostModel,
};
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};

use std::sync::Arc;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::idle::IdleManager;
use crate::proxy::ModelRegistry;
use crate::server::ModalBackend;

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
        uint32 unitsConsumed;
        string outputType;
        uint32 latencyMs;
    }
}

pub const INFERENCE_JOB: u8 = 0;

/// Tangle job router — single generic inference job.
/// The `model` field in the request selects the Modal endpoint.
pub fn router() -> Router {
    Router::new().route(
        INFERENCE_JOB,
        run_inference
            .layer(TangleLayer)
            .layer(blueprint_sdk::tee::TeeLayer::new()),
    )
}

/// Generic inference job handler.
///
/// On-chain jobs exist as a billing-verification path; the real traffic flows
/// through the OpenAI-compatible HTTP proxy. This handler echoes metadata so
/// the blueprint is valid on Tangle and can receive on-chain job calls.
#[debug_job]
pub async fn run_inference(
    TangleArg(req): TangleArg<InferenceRequest>,
) -> TangleResult<InferenceResult> {
    TangleResult(InferenceResult {
        outputData: Vec::new().into(),
        unitsConsumed: 0,
        outputType: req.outputType,
        latencyMs: 0,
    })
}

// ---------------------------------------------------------------------------
// Background service: HTTP proxy server
// ---------------------------------------------------------------------------

/// BackgroundService wrapper that builds the AppState and launches the
/// Axum HTTP proxy for Modal.
pub struct ModalInferenceServer {
    pub config: Arc<OperatorConfig>,
}

impl ModalInferenceServer {
    pub fn new(config: OperatorConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

impl BackgroundService for ModalInferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            let registry = ModelRegistry::new(config.modal.models.clone());

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
                    check_mins = config.modal.idle_check_interval_minutes,
                    "Idle shutdown enabled"
                );
                Some(mgr)
            } else {
                None
            };

            let billing_client = match BillingClient::new(&config.tangle, &config.billing) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    tracing::error!(error = %e, "failed to create billing client");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            let operator_address = billing_client.operator_address();
            let nonce_store = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));
            let backend = ModalBackend::new(config.clone(), registry, idle_mgr);

            let state = match AppStateBuilder::new()
                .billing(billing_client)
                .nonce_store(nonce_store)
                .server_config(Arc::new(config.server.clone()))
                .billing_config(Arc::new(config.billing.clone()))
                .tangle_config(Arc::new(config.tangle.clone()))
                .operator_address(operator_address)
                .backend(backend)
                .build()
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to build AppState");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            match server::start(state, shutdown_rx).await {
                Ok(_handle) => {
                    tracing::info!("Modal HTTP server started — background service ready");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                }
            }
        });

        Ok(rx)
    }
}
