pub mod config;
pub mod proxy;
pub use tangle_inference_core::qos;
pub mod server;

// Re-export shared infrastructure so downstream crates can
// `use modal_inference::*`.
pub use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};
pub use tangle_inference_core::{
    billing, metrics, AppState, AppStateBuilder, BillingClient, CostModel, CostParams,
    FlatRequestCostModel, NonceStore, PerCharCostModel, PerImageCostModel, PerSecondCostModel,
    PerTokenCostModel, RequestGuard, SpendAuthPayload, TaskTypeCostModel,
};

use std::sync::Arc;

use alloy_sol_types::sol;
use blueprint_sdk::extract::Context;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::proxy::ModelRegistry;
use crate::server::ModalBackend;

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
pub fn router_with_config(config: Arc<OperatorConfig>) -> Router {
    Router::new()
        .route(
            INFERENCE_JOB,
            run_inference
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .with_context(Arc::new(ModelRegistry::new(config.modal.models.clone())))
}

#[debug_job]
pub async fn run_inference(
    Context(registry): Context<Arc<ModelRegistry>>,
    TangleArg(req): TangleArg<InferenceRequest>,
) -> anyhow::Result<TangleResult<InferenceResult>> {
    let model = registry
        .get(&req.model)
        .ok_or_else(|| anyhow::anyhow!("model not configured"))?;
    let mut input_units = 0;
    let (body, content_type) = match req.inputType.as_str() {
        "audio" | "audio/wav" => (bytes::Bytes::from(req.inputData.to_vec()), "audio/wav"),
        "json" | "application/json" => {
            serde_json::from_slice::<serde_json::Value>(&req.inputData)?;
            (
                bytes::Bytes::from(req.inputData.to_vec()),
                "application/json",
            )
        }
        "text" => {
            let text = std::str::from_utf8(&req.inputData)?;
            let value = if let Some(value) = serde_json::from_str::<serde_json::Value>(text)
                .ok()
                .filter(serde_json::Value::is_object)
            {
                value
            } else {
                match model.task_type.as_str() {
                    "tts" => {
                        input_units = u32::try_from(text.chars().count())?;
                        serde_json::json!({"text":text,"voice_id":"default","format":"wav"})
                    }
                    "chat" | "text" | "text-generation" => {
                        serde_json::json!({"model":req.model,
                        "messages":[{"role":"user","content":text}],"stream":false})
                    }
                    "embedding" => serde_json::json!({"model":req.model,"input":text}),
                    _ => serde_json::json!({"prompt":text}),
                }
            };
            (
                bytes::Bytes::from(serde_json::to_vec(&value)?),
                "application/json",
            )
        }
        _ => anyhow::bail!("unsupported inputType; use text, json or audio"),
    };
    let output = registry
        .proxy_request(&req.model, None, body, content_type)
        .await?;
    anyhow::ensure!(
        !output.data.is_empty(),
        "provider returned an empty inference result"
    );
    let units = serde_json::from_slice::<serde_json::Value>(&output.data)
        .ok()
        .and_then(|value| {
            value["usage"]["total_tokens"]
                .as_u64()
                .or(value["units_consumed"].as_u64())
        })
        .map(u32::try_from)
        .transpose()?
        .unwrap_or(input_units);
    Ok(TangleResult(InferenceResult {
        outputData: output.data.to_vec().into(),
        unitsConsumed: units,
        outputType: req.outputType,
        latencyMs: u32::try_from(output.latency_ms)?,
    }))
}

/// BackgroundService wrapper that builds the AppState and launches the
/// Axum HTTP proxy for Modal.
pub struct ModalInferenceServer {
    pub config: Arc<OperatorConfig>,
}

pub fn app_state(config: Arc<OperatorConfig>) -> anyhow::Result<AppState> {
    let billing = Arc::new(BillingClient::new(&config.tangle, &config.billing)?);
    let operator = billing.operator_address();
    let nonces = Arc::new(NonceStore::load(config.billing.nonce_store_path.clone()));
    let registry = ModelRegistry::new(config.modal.models.clone());
    AppStateBuilder::new()
        .billing(billing)
        .nonce_store(nonces)
        .server_config(Arc::new(config.server.clone()))
        .billing_config(Arc::new(config.billing.clone()))
        .tangle_config(Arc::new(config.tangle.clone()))
        .operator_address(operator)
        .backend(ModalBackend::new(config, registry))
        .build()
}

impl BackgroundService for ModalInferenceServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();
        tokio::spawn(async move {
            let state = match app_state(config) {
                Ok(state) => state,
                Err(error) => {
                    let _ = tx.send(Err(RunnerError::Other(error.to_string().into())));
                    return;
                }
            };
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let result = match server::start(state, shutdown_rx).await {
                Ok(handle) => handle
                    .await
                    .map_err(|error| RunnerError::Other(error.to_string().into())),
                Err(error) => Err(RunnerError::Other(error.to_string().into())),
            };
            drop(shutdown_tx);
            let _ = tx.send(result);
        });
        Ok(rx)
    }
}
