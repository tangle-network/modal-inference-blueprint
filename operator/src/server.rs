//! Axum HTTP server — OpenAI-compatible proxy to Modal deployments.
//!
//! All shared infrastructure (nonce store, spend-auth validation, x402 headers,
//! metrics, app state container) lives in `tangle-inference-core`. This module
//! contains the modal-specific backend struct, OpenAI-compatible request
//! handlers, and task-type aware cost calculation via `TaskTypeCostModel`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Json, Multipart, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use serde::Deserialize;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;

use tangle_inference_core::server::{billing_gate, error_response, metrics_handler, settle_billing};
use tangle_inference_core::{
    AppState, CostModel, CostParams, FlatRequestCostModel, PerCharCostModel, PerImageCostModel,
    PerSecondCostModel, PerTokenCostModel, SpendAuthPayload, TaskTypeCostModel,
};

use crate::config::{ModelEndpoint, OperatorConfig};
use crate::idle::IdleManager;
use crate::proxy::ModelRegistry;

// ---------------------------------------------------------------------------
// ModalBackend — attached to AppState via AppStateBuilder
// ---------------------------------------------------------------------------

/// Backend state for the Modal proxy. Holds the model registry, idle manager,
/// pre-built task-type cost model, and a reference to the full operator config
/// (for modal-specific knobs only — all shared knobs live on `AppState`).
pub struct ModalBackend {
    pub config: Arc<OperatorConfig>,
    pub registry: ModelRegistry,
    pub idle_manager: Option<Arc<IdleManager>>,
    pub cost_model: Arc<TaskTypeCostModel>,
}

impl ModalBackend {
    pub fn new(
        config: Arc<OperatorConfig>,
        registry: ModelRegistry,
        idle_manager: Option<Arc<IdleManager>>,
    ) -> Self {
        let modal = &config.modal;

        let mut per_task: HashMap<String, Box<dyn CostModel>> = HashMap::new();
        per_task.insert(
            "chat".into(),
            Box::new(PerTokenCostModel {
                price_per_input_token: modal.price_per_input_token,
                price_per_output_token: modal.price_per_output_token,
            }),
        );
        per_task.insert(
            "tts".into(),
            Box::new(PerCharCostModel {
                price_per_1k_chars: modal.price_per_1k_tts_chars,
            }),
        );
        per_task.insert(
            "stt".into(),
            Box::new(PerSecondCostModel {
                price_per_second: modal.price_per_stt_second,
            }),
        );
        per_task.insert(
            "image".into(),
            Box::new(PerImageCostModel {
                price_per_image: modal.price_per_image,
            }),
        );
        per_task.insert(
            "video".into(),
            Box::new(PerSecondCostModel {
                price_per_second: modal.price_per_video_second,
            }),
        );
        per_task.insert(
            "music".into(),
            Box::new(PerSecondCostModel {
                price_per_second: modal.price_per_music_second,
            }),
        );
        per_task.insert(
            "embedding".into(),
            Box::new(PerTokenCostModel {
                price_per_input_token: modal.price_per_1k_embedding_tokens,
                price_per_output_token: 0,
            }),
        );

        let cost_model = Arc::new(TaskTypeCostModel {
            default: Box::new(FlatRequestCostModel {
                price_per_request: modal.default_price_per_request,
            }),
            per_task,
        });

        Self {
            config,
            registry,
            idle_manager,
            cost_model,
        }
    }

    /// Calculate cost for a given task type and usage.
    pub fn calculate_cost(&self, params: &CostParams) -> u64 {
        self.cost_model.calculate_cost(params)
    }
}

/// Map a model's verbose task_type (e.g. "text-generation", "video-avatar") to
/// the canonical cost-model key ("chat", "tts", "stt", "image", "video",
/// "music", "embedding"). Unknown values return `None`, which makes the
/// `TaskTypeCostModel` fall through to the default (flat per-request) model.
fn canonical_task_key(task_type: &str) -> Option<&'static str> {
    match task_type {
        "tts" | "clone" | "voice-design" => Some("tts"),
        "stt" | "diarize" | "translate" | "langid" | "vad" | "s2s" | "speakerid" => Some("stt"),
        "video-generation" | "video" | "video-avatar" | "video-lipsync" | "video-understanding" => {
            Some("video")
        }
        "image-generation" | "image" => Some("image"),
        "music-generation" | "music" => Some("music"),
        "chat" | "text-generation" | "text" => Some("chat"),
        "embedding" | "rerank" => Some("embedding"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Router wiring
// ---------------------------------------------------------------------------

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(synthesize))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/videos/generations", post(videos_generations))
        .route("/v1/embeddings", post(embeddings))
        .route("/proxy/:model/*path", post(proxy_raw))
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/models", get(list_models))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TimeoutLayer::new(Duration::from_secs(120)))
}

/// Start the HTTP server with graceful shutdown, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let backend = state
        .backend::<ModalBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not a ModalBackend"))?;
    let bind = format!(
        "{}:{}",
        state.server_config.host, state.server_config.port
    );
    let _ = backend;

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "Modal HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

fn backend_from(state: &AppState) -> &ModalBackend {
    state
        .backend::<ModalBackend>()
        .expect("AppState backend is ModalBackend")
}

async fn ensure_model_awake(backend: &ModalBackend, model_name: &str) -> Result<(), Response> {
    if let Some(ref mgr) = backend.idle_manager {
        if !mgr.record_request(model_name).await {
            if let Err(e) = mgr.wake_model(model_name).await {
                return Err(error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Model waking: {e}"),
                    "upstream_error",
                    "wake_failed",
                ));
            }
        }
    }
    Ok(())
}

fn resolve_model<'a>(
    backend: &'a ModalBackend,
    name: Option<&str>,
    fallback_task: &str,
) -> Option<&'a ModelEndpoint> {
    name.and_then(|n| backend.registry.get(n))
        .or_else(|| backend.registry.list_by_type(fallback_task).into_iter().next())
}

/// Wake model, proxy request, settle billing, return response.
/// `actual_cost_params` lets the caller override the settle cost (e.g. when
/// actual usage is known from the response). When `None`, `estimated_cost` is
/// used for settlement.
async fn proxy_and_settle(
    state: &AppState,
    backend: &ModalBackend,
    model_name: &str,
    path: Option<&str>,
    payload: Bytes,
    content_type: &str,
    spend_auth: &Option<SpendAuthPayload>,
    preauth_amount: Option<u64>,
    actual_cost: u64,
) -> Response {
    if let Err(r) = ensure_model_awake(backend, model_name).await {
        return r;
    }

    match backend
        .registry
        .proxy_request(model_name, path, payload, content_type)
        .await
    {
        Ok(resp) => {
            if let (Some(ref sa), Some(preauth)) = (spend_auth, preauth_amount) {
                if let Err(e) = settle_billing(&state.billing, sa, preauth, actual_cost).await {
                    tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
                }
            }
            (
                StatusCode::OK,
                [("content-type", resp.content_type.as_str())],
                resp.data,
            )
                .into_response()
        }
        Err(e) => error_response(
            StatusCode::BAD_GATEWAY,
            format!("Proxy error: {e}"),
            "upstream_error",
            "modal_error",
        ),
    }
}

fn json_payload(value: &serde_json::Value) -> Result<Bytes, Response> {
    serde_json::to_vec(value).map(Bytes::from).map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to serialize payload: {e}"),
            "internal_error",
            "serialize_failed",
        )
    })
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SpeechRequest {
    model: Option<String>,
    input: String,
    voice: Option<String>,
    response_format: Option<String>,
    spend_auth: Option<SpendAuthPayload>,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    spend_auth: Option<SpendAuthPayload>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ImageGenerationRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_image_n")]
    n: u32,
    spend_auth: Option<SpendAuthPayload>,
}

#[derive(Deserialize)]
struct VideoGenerationRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_video_duration")]
    duration_seconds: u32,
    spend_auth: Option<SpendAuthPayload>,
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    model: Option<String>,
    input: serde_json::Value,
    spend_auth: Option<SpendAuthPayload>,
}

fn default_max_tokens() -> u32 {
    512
}
fn default_temperature() -> f32 {
    0.7
}
fn default_image_n() -> u32 {
    1
}
fn default_video_duration() -> u32 {
    5
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/audio/speech — OpenAI-compatible TTS
async fn synthesize(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SpeechRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "tts") else {
        return error_response(StatusCode::BAD_REQUEST, "No TTS model configured".into(), "invalid_request_error", "no_model");
    };

    let chars = body.input.len() as u64;
    let cost_params = CostParams {
        task_type: Some("tts".into()),
        extra: HashMap::from([("characters".into(), chars)]),
        ..Default::default()
    };
    let estimated_cost = backend.calculate_cost(&CostParams {
        extra: HashMap::from([("characters".into(), chars.max(500))]),
        ..cost_params.clone()
    });

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body.spend_auth, estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    let payload = serde_json::json!({
        "text": body.input,
        "voice_id": body.voice.as_deref().unwrap_or("default"),
        "format": body.response_format.as_deref().unwrap_or("wav"),
    });
    let payload_bytes = match json_payload(&payload) {
        Ok(b) => b,
        Err(r) => return r,
    };

    proxy_and_settle(
        &state, backend, &model.name, None, payload_bytes, "application/json",
        &spend_auth, preauth_amount, backend.calculate_cost(&cost_params),
    )
    .await
}

/// POST /v1/audio/transcriptions — OpenAI-compatible STT
async fn transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let backend = backend_from(&state);

    let mut audio_data: Option<Bytes> = None;
    let mut model_name_field = "default".to_string();
    let mut spend_auth_json: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("file") => audio_data = field.bytes().await.ok(),
            Some("model") => model_name_field = field.text().await.unwrap_or_default(),
            Some("spend_auth") => spend_auth_json = field.text().await.ok(),
            _ => {}
        }
    }

    let Some(audio) = audio_data else {
        return error_response(StatusCode::BAD_REQUEST, "Missing audio file".into(), "invalid_request_error", "missing_file");
    };
    let Some(model) = resolve_model(backend, Some(&model_name_field), "stt") else {
        return error_response(StatusCode::BAD_REQUEST, "No STT model configured".into(), "invalid_request_error", "no_model");
    };

    // 16kHz mono 16-bit PCM = 32,000 bytes/sec
    let centiseconds = (audio.len() as u64 * 100) / 32000;
    let cost_params = CostParams {
        task_type: Some("stt".into()),
        extra: HashMap::from([("centiseconds".into(), centiseconds)]),
        ..Default::default()
    };
    let estimated_cost = backend.calculate_cost(&CostParams {
        extra: HashMap::from([("centiseconds".into(), centiseconds.max(3000))]),
        ..cost_params.clone()
    });

    let body_spend_auth: Option<SpendAuthPayload> =
        spend_auth_json.as_deref().and_then(|s| serde_json::from_str(s).ok());

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body_spend_auth, estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    proxy_and_settle(
        &state, backend, &model.name, None, audio, "audio/wav",
        &spend_auth, preauth_amount, backend.calculate_cost(&cost_params),
    )
    .await
}

/// POST /v1/chat/completions — needs custom settlement (parses usage from response).
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "chat") else {
        return error_response(StatusCode::BAD_REQUEST, "No chat model configured".into(), "invalid_request_error", "no_model");
    };

    let est_prompt: u32 = body.messages.iter().map(|m| (m.content.len() as u32) / 4 + 1).sum();
    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("chat".into()),
        prompt_tokens: est_prompt,
        completion_tokens: body.max_tokens,
        ..Default::default()
    });

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body.spend_auth, estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    if let Err(r) = ensure_model_awake(backend, &model.name).await {
        return r;
    }

    let payload = serde_json::json!({
        "model": model.name,
        "messages": body.messages.iter().map(|m| serde_json::json!({"role": m.role, "content": m.content})).collect::<Vec<_>>(),
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    });
    let payload_bytes = match json_payload(&payload) {
        Ok(b) => b,
        Err(r) => return r,
    };

    match backend.registry.proxy_request(&model.name, None, payload_bytes, "application/json").await {
        Ok(resp) => {
            // Parse actual usage from response for accurate settlement.
            let (pt, ct) = serde_json::from_slice::<serde_json::Value>(&resp.data)
                .ok()
                .and_then(|v| {
                    let u = v.get("usage")?;
                    Some((
                        u.get("prompt_tokens")?.as_u64()? as u32,
                        u.get("completion_tokens")?.as_u64()? as u32,
                    ))
                })
                .unwrap_or((est_prompt, body.max_tokens));

            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("chat".into()),
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
                    tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
                }
            }

            (StatusCode::OK, [("content-type", resp.content_type.as_str())], resp.data).into_response()
        }
        Err(e) => error_response(StatusCode::BAD_GATEWAY, format!("Proxy error: {e}"), "upstream_error", "modal_error"),
    }
}

/// POST /v1/images/generations
async fn images_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ImageGenerationRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "image") else {
        return error_response(StatusCode::BAD_REQUEST, "No image model configured".into(), "invalid_request_error", "no_model");
    };

    let cost = backend.calculate_cost(&CostParams {
        task_type: Some("image".into()),
        extra: HashMap::from([("images".into(), body.n.max(1) as u64)]),
        ..Default::default()
    });

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body.spend_auth, cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    let payload = serde_json::json!({"model": model.name, "prompt": body.prompt, "n": body.n});
    let payload_bytes = match json_payload(&payload) { Ok(b) => b, Err(r) => return r };

    proxy_and_settle(
        &state, backend, &model.name, None, payload_bytes, "application/json",
        &spend_auth, preauth_amount, cost,
    ).await
}

/// POST /v1/videos/generations
async fn videos_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<VideoGenerationRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "video") else {
        return error_response(StatusCode::BAD_REQUEST, "No video model configured".into(), "invalid_request_error", "no_model");
    };

    let cost = backend.calculate_cost(&CostParams {
        task_type: Some("video".into()),
        extra: HashMap::from([("centiseconds".into(), (body.duration_seconds as u64) * 100)]),
        ..Default::default()
    });

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body.spend_auth, cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    let payload = serde_json::json!({"model": model.name, "prompt": body.prompt, "duration_seconds": body.duration_seconds});
    let payload_bytes = match json_payload(&payload) { Ok(b) => b, Err(r) => return r };

    proxy_and_settle(
        &state, backend, &model.name, None, payload_bytes, "application/json",
        &spend_auth, preauth_amount, cost,
    ).await
}

/// POST /v1/embeddings
async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "embedding") else {
        return error_response(StatusCode::BAD_REQUEST, "No embedding model configured".into(), "invalid_request_error", "no_model");
    };

    let approx_chars: u64 = match &body.input {
        serde_json::Value::String(s) => s.len() as u64,
        serde_json::Value::Array(arr) => arr.iter().filter_map(|v| v.as_str().map(|s| s.len() as u64)).sum(),
        _ => 0,
    };
    let cost = backend.calculate_cost(&CostParams {
        task_type: Some("embedding".into()),
        prompt_tokens: (approx_chars / 4).max(1) as u32,
        ..Default::default()
    });

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body.spend_auth, cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    let payload = serde_json::json!({"model": model.name, "input": body.input});
    let payload_bytes = match json_payload(&payload) { Ok(b) => b, Err(r) => return r };

    proxy_and_settle(
        &state, backend, &model.name, None, payload_bytes, "application/json",
        &spend_auth, preauth_amount, cost,
    ).await
}

/// POST /proxy/:model/*path — raw proxy to any model endpoint.
/// SpendAuth via X-Payment-Signature. Task-type drives cost via `TaskTypeCostModel`.
async fn proxy_raw(
    State(state): State<AppState>,
    Path((model, path)): Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let backend = backend_from(&state);
    let task_key = backend.registry.get(&model)
        .and_then(|m| canonical_task_key(&m.task_type));

    let settle_cost = backend.calculate_cost(&CostParams {
        task_type: task_key.map(String::from),
        extra: HashMap::from([("images".into(), 1)]),
        ..Default::default()
    });
    let estimated_cost = if task_key.is_some() {
        backend.calculate_cost(&CostParams {
            task_type: task_key.map(String::from),
            extra: HashMap::from([("images".into(), 1), ("centiseconds".into(), 100), ("characters".into(), 500)]),
            prompt_tokens: 500,
            completion_tokens: 500,
        })
    } else {
        settle_cost
    };

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, None, estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    let proxy_path = format!("/{path}");
    proxy_and_settle(
        &state, backend, &model, Some(&proxy_path), body, "application/json",
        &spend_auth, preauth_amount, settle_cost,
    ).await
}

// ---------------------------------------------------------------------------
// Read-only endpoints
// ---------------------------------------------------------------------------

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let models: Vec<serde_json::Value> = backend
        .registry
        .list()
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "type": m.task_type,
                "endpoint": m.modal_endpoint,
                "object": "model",
            })
        })
        .collect();

    Json(serde_json::json!({ "object": "list", "data": models }))
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let model_health = backend.registry.health_check_all().await;
    let all_ok = model_health.iter().all(|h| h.status == "ok");

    let models: Vec<serde_json::Value> = model_health
        .iter()
        .map(|h| {
            let model_config = backend.registry.get(&h.name);
            serde_json::json!({
                "name": h.name,
                "status": h.status,
                "type": model_config.map(|m| m.task_type.as_str()).unwrap_or("unknown"),
                "latency_ms": h.latency_ms,
                "modal_endpoint": model_config.map(|m| m.modal_endpoint.as_str()),
            })
        })
        .collect();

    Json(serde_json::json!({
        "status": if all_ok { "ok" } else { "degraded" },
        "operator": backend.config.name,
        "billing_required": state.billing_config.billing_required,
        "models": models,
        "metrics": tangle_inference_core::metrics::health_summary(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TangleConfig;

    #[test]
    fn test_canonical_task_key() {
        assert_eq!(canonical_task_key("tts"), Some("tts"));
        assert_eq!(canonical_task_key("clone"), Some("tts"));
        assert_eq!(canonical_task_key("stt"), Some("stt"));
        assert_eq!(canonical_task_key("diarize"), Some("stt"));
        assert_eq!(canonical_task_key("video-generation"), Some("video"));
        assert_eq!(canonical_task_key("image-generation"), Some("image"));
        assert_eq!(canonical_task_key("music"), Some("music"));
        assert_eq!(canonical_task_key("chat"), Some("chat"));
        assert_eq!(canonical_task_key("text-generation"), Some("chat"));
        assert_eq!(canonical_task_key("embedding"), Some("embedding"));
        assert_eq!(canonical_task_key("unknown-xyz"), None);
    }

    #[test]
    fn test_modal_backend_cost_dispatch() {
        let config = Arc::new(OperatorConfig {
            name: "t".into(),
            tangle: TangleConfig_for_test(),
            server: serde_json::from_str("{}").unwrap(),
            billing: serde_json::from_str(r#"{"max_spend_per_request":0,"min_credit_balance":0}"#)
                .unwrap(),
            modal: crate::config::ModalConfig {
                price_per_input_token: 2,
                price_per_output_token: 5,
                price_per_1k_tts_chars: 10_000,
                price_per_stt_second: 100,
                price_per_image: 500,
                price_per_video_second: 1000,
                price_per_music_second: 50,
                price_per_1k_embedding_tokens: 1,
                default_price_per_request: 42,
                models: vec![],
                idle_shutdown_minutes: 0,
                idle_check_interval_minutes: 0,
            },
            qos: None,
        });

        let registry = ModelRegistry::new(vec![]);
        let backend = ModalBackend::new(config, registry, None);

        // chat
        let cost = backend.calculate_cost(&CostParams {
            task_type: Some("chat".into()),
            prompt_tokens: 100,
            completion_tokens: 50,
            ..Default::default()
        });
        assert_eq!(cost, 100 * 2 + 50 * 5);

        // tts
        let cost = backend.calculate_cost(&CostParams {
            task_type: Some("tts".into()),
            extra: HashMap::from([("characters".into(), 2000)]),
            ..Default::default()
        });
        assert_eq!(cost, (2000 * 10_000) / 1000);

        // stt
        let cost = backend.calculate_cost(&CostParams {
            task_type: Some("stt".into()),
            extra: HashMap::from([("centiseconds".into(), 500)]),
            ..Default::default()
        });
        assert_eq!(cost, (500 * 100) / 100);

        // image
        let cost = backend.calculate_cost(&CostParams {
            task_type: Some("image".into()),
            extra: HashMap::from([("images".into(), 3)]),
            ..Default::default()
        });
        assert_eq!(cost, 3 * 500);

        // unknown → default flat
        let cost = backend.calculate_cost(&CostParams {
            task_type: Some("unknown".into()),
            ..Default::default()
        });
        assert_eq!(cost, 42);
    }

    #[allow(non_snake_case)]
    fn TangleConfig_for_test() -> TangleConfig {
        serde_json::from_str(
            r#"{
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0x0000000000000000000000000000000000000000000000000000000000000000",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1
            }"#,
        )
        .unwrap()
    }
}
