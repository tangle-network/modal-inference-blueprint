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
    body::Body,
    extract::{Json, Multipart, Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use serde::Deserialize;
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;

use tangle_inference_core::server::{
    error_response, extract_x402_spend_auth, payment_required, settle_billing, validate_spend_auth,
};
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
        .route("/metrics", get(prom_metrics))
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

// ---------------------------------------------------------------------------
// Shared billing gate for task-aware handlers
// ---------------------------------------------------------------------------

/// Handles the full billing gate: extract SpendAuth (body > x402 header),
/// validate it, and pre-authorize on-chain. Returns the pre-auth amount or
/// an error response.
async fn billing_gate(
    state: &AppState,
    headers: &HeaderMap,
    body_spend_auth: Option<SpendAuthPayload>,
    task_key: Option<&str>,
    estimated_cost: u64,
) -> Result<(Option<SpendAuthPayload>, Option<u64>), Response> {
    let spend_auth = body_spend_auth.or_else(|| extract_x402_spend_auth(headers));

    if !state.billing_config.billing_required {
        return Ok((spend_auth, None));
    }

    let Some(spend_auth) = spend_auth else {
        let _ = task_key;
        return Err(payment_required(
            &state.billing_config,
            &state.tangle_config,
            state.operator_address,
            estimated_cost.max(state.billing_config.min_charge_amount),
        ));
    };

    let preauth_amount = match validate_spend_auth(state, &spend_auth).await {
        Ok(amt) => amt,
        Err(resp) => return Err(resp),
    };

    if let Err(e) = state.billing.authorize_spend(&spend_auth).await {
        tracing::error!(error = %e, "authorizeSpend failed");
        return Err(error_response(
            StatusCode::PAYMENT_REQUIRED,
            format!("billing authorization failed: {e}"),
            "billing_error",
            "authorization_failed",
        ));
    }

    // NOTE: validate_spend_auth records the nonce internally — no separate insert needed.

    Ok((Some(spend_auth), Some(preauth_amount)))
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
        return error_response(
            StatusCode::BAD_REQUEST,
            "No TTS model configured".into(),
            "invalid_request_error",
            "no_model",
        );
    };

    let chars = body.input.len() as u64;
    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("tts".into()),
        extra: HashMap::from([("characters".into(), chars.max(500))]),
        ..Default::default()
    });

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body.spend_auth, Some("tts"), estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    if let Err(r) = ensure_model_awake(backend, &model.name).await {
        return r;
    }

    let payload = serde_json::json!({
        "text": body.input,
        "voice_id": body.voice.as_deref().unwrap_or("default"),
        "format": body.response_format.as_deref().unwrap_or("wav"),
    });
    let payload_bytes = match serde_json::to_vec(&payload) {
        Ok(b) => Bytes::from(b),
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to serialize TTS payload: {e}"),
                "internal_error",
                "serialize_failed",
            );
        }
    };

    match backend
        .registry
        .proxy_request(&model.name, None, payload_bytes, "application/json")
        .await
    {
        Ok(resp) => {
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("tts".into()),
                    extra: HashMap::from([("characters".into(), chars)]),
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
                    tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
                }
            }

            (
                StatusCode::OK,
                [
                    ("content-type", resp.content_type.as_str()),
                    ("x-model", &model.name),
                    ("x-latency-ms", &resp.latency_ms.to_string()),
                ],
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
            Some("file") => {
                audio_data = field.bytes().await.ok();
            }
            Some("model") => {
                model_name_field = field.text().await.unwrap_or_default();
            }
            Some("spend_auth") => {
                spend_auth_json = field.text().await.ok();
            }
            _ => {}
        }
    }

    let Some(audio) = audio_data else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Missing audio file".into(),
            "invalid_request_error",
            "missing_file",
        );
    };

    let Some(model) = resolve_model(backend, Some(&model_name_field), "stt") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "No STT model configured".into(),
            "invalid_request_error",
            "no_model",
        );
    };

    let audio_len = audio.len();
    // 16kHz mono 16-bit PCM = 32,000 bytes/sec
    let centiseconds = (audio_len as u64 * 100) / 32000;

    let body_spend_auth: Option<SpendAuthPayload> =
        spend_auth_json.as_deref().and_then(|s| serde_json::from_str(s).ok());

    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("stt".into()),
        extra: HashMap::from([("centiseconds".into(), centiseconds.max(3000))]),
        ..Default::default()
    });

    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, body_spend_auth, Some("stt"), estimated_cost).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    if let Err(r) = ensure_model_awake(backend, &model.name).await {
        return r;
    }

    match backend
        .registry
        .proxy_request(&model.name, None, audio, "audio/wav")
        .await
    {
        Ok(resp) => {
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("stt".into()),
                    extra: HashMap::from([("centiseconds".into(), centiseconds)]),
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
                    tracing::error!(error = %e, "on-chain settlement failed — manual recovery required");
                }
            }

            (
                StatusCode::OK,
                [("content-type", "application/json")],
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

/// POST /v1/chat/completions
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "chat") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "No chat model configured".into(),
            "invalid_request_error",
            "no_model",
        );
    };

    let estimated_prompt_tokens: u32 = body
        .messages
        .iter()
        .map(|m| (m.content.len() as u32) / 4 + 1)
        .sum();
    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("chat".into()),
        prompt_tokens: estimated_prompt_tokens,
        completion_tokens: body.max_tokens,
        ..Default::default()
    });

    let (spend_auth, preauth_amount) = match billing_gate(
        &state,
        &headers,
        body.spend_auth,
        Some("chat"),
        estimated_cost,
    )
    .await
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if let Err(r) = ensure_model_awake(backend, &model.name).await {
        return r;
    }

    let payload = serde_json::json!({
        "model": model.name,
        "messages": body.messages.iter().map(|m| serde_json::json!({
            "role": m.role,
            "content": m.content,
        })).collect::<Vec<_>>(),
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
    });
    let payload_bytes = match serde_json::to_vec(&payload) {
        Ok(b) => Bytes::from(b),
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to serialize chat payload: {e}"),
                "internal_error",
                "serialize_failed",
            );
        }
    };

    match backend
        .registry
        .proxy_request(&model.name, None, payload_bytes, "application/json")
        .await
    {
        Ok(resp) => {
            // Parse usage if present.
            let (prompt_tokens, completion_tokens) =
                match serde_json::from_slice::<serde_json::Value>(&resp.data) {
                    Ok(v) => {
                        let pt =
                            v.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|x| x.as_u64())
                                .unwrap_or(estimated_prompt_tokens as u64) as u32;
                        let ct =
                            v.get("usage").and_then(|u| u.get("completion_tokens")).and_then(|x| x.as_u64())
                                .unwrap_or(body.max_tokens as u64) as u32;
                        (pt, ct)
                    }
                    Err(_) => (estimated_prompt_tokens, body.max_tokens),
                };

            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("chat".into()),
                    prompt_tokens,
                    completion_tokens,
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
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

/// POST /v1/images/generations
async fn images_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ImageGenerationRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "image") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "No image model configured".into(),
            "invalid_request_error",
            "no_model",
        );
    };

    let images = body.n.max(1) as u64;
    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("image".into()),
        extra: HashMap::from([("images".into(), images)]),
        ..Default::default()
    });

    let (spend_auth, preauth_amount) = match billing_gate(
        &state,
        &headers,
        body.spend_auth,
        Some("image"),
        estimated_cost,
    )
    .await
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if let Err(r) = ensure_model_awake(backend, &model.name).await {
        return r;
    }

    let payload = serde_json::json!({
        "model": model.name,
        "prompt": body.prompt,
        "n": body.n,
    });
    let payload_bytes = Bytes::from(serde_json::to_vec(&payload).unwrap_or_default());

    match backend
        .registry
        .proxy_request(&model.name, None, payload_bytes, "application/json")
        .await
    {
        Ok(resp) => {
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("image".into()),
                    extra: HashMap::from([("images".into(), images)]),
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
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

/// POST /v1/videos/generations
async fn videos_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<VideoGenerationRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "video") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "No video model configured".into(),
            "invalid_request_error",
            "no_model",
        );
    };

    let centiseconds = (body.duration_seconds as u64) * 100;
    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("video".into()),
        extra: HashMap::from([("centiseconds".into(), centiseconds)]),
        ..Default::default()
    });

    let (spend_auth, preauth_amount) = match billing_gate(
        &state,
        &headers,
        body.spend_auth,
        Some("video"),
        estimated_cost,
    )
    .await
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if let Err(r) = ensure_model_awake(backend, &model.name).await {
        return r;
    }

    let payload = serde_json::json!({
        "model": model.name,
        "prompt": body.prompt,
        "duration_seconds": body.duration_seconds,
    });
    let payload_bytes = Bytes::from(serde_json::to_vec(&payload).unwrap_or_default());

    match backend
        .registry
        .proxy_request(&model.name, None, payload_bytes, "application/json")
        .await
    {
        Ok(resp) => {
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("video".into()),
                    extra: HashMap::from([("centiseconds".into(), centiseconds)]),
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
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

/// POST /v1/embeddings
async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    let backend = backend_from(&state);
    let Some(model) = resolve_model(backend, body.model.as_deref(), "embedding") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "No embedding model configured".into(),
            "invalid_request_error",
            "no_model",
        );
    };

    // Estimate tokens as characters/4 across all inputs.
    let approx_chars: u64 = match &body.input {
        serde_json::Value::String(s) => s.len() as u64,
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.len() as u64))
            .sum(),
        _ => 0,
    };
    let approx_tokens = (approx_chars / 4).max(1) as u32;

    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("embedding".into()),
        prompt_tokens: approx_tokens,
        ..Default::default()
    });

    let (spend_auth, preauth_amount) = match billing_gate(
        &state,
        &headers,
        body.spend_auth,
        Some("embedding"),
        estimated_cost,
    )
    .await
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if let Err(r) = ensure_model_awake(backend, &model.name).await {
        return r;
    }

    let payload = serde_json::json!({
        "model": model.name,
        "input": body.input,
    });
    let payload_bytes = Bytes::from(serde_json::to_vec(&payload).unwrap_or_default());

    match backend
        .registry
        .proxy_request(&model.name, None, payload_bytes, "application/json")
        .await
    {
        Ok(resp) => {
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("embedding".into()),
                    prompt_tokens: approx_tokens,
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
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

/// POST /proxy/:model/*path — raw proxy to any model endpoint.
///
/// SpendAuth is expected via X-Payment-Signature. The model's declared
/// task_type drives cost calculation via `TaskTypeCostModel`.
async fn proxy_raw(
    State(state): State<AppState>,
    Path((model, path)): Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let backend = backend_from(&state);
    let model_cfg = backend.registry.get(&model);
    let task_key = model_cfg
        .and_then(|m| canonical_task_key(&m.task_type))
        .unwrap_or("");

    let estimated_cost = if task_key.is_empty() {
        backend.calculate_cost(&CostParams::default())
    } else {
        backend.calculate_cost(&CostParams {
            task_type: Some(task_key.to_string()),
            extra: HashMap::from([
                ("images".into(), 1),
                ("centiseconds".into(), 100),
                ("characters".into(), 500),
            ]),
            prompt_tokens: 500,
            completion_tokens: 500,
        })
    };

    let (spend_auth, preauth_amount) = match billing_gate(
        &state,
        &headers,
        None,
        Some(task_key),
        estimated_cost,
    )
    .await
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    if let Err(r) = ensure_model_awake(backend, &model).await {
        return r;
    }

    let proxy_path = format!("/{path}");
    match backend
        .registry
        .proxy_request(&model, Some(&proxy_path), body, "application/json")
        .await
    {
        Ok(resp) => {
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                // Conservative settle: charge the default request cost at minimum,
                // or the task-type cost model's flat interpretation.
                let cost = backend.calculate_cost(&CostParams {
                    task_type: if task_key.is_empty() {
                        None
                    } else {
                        Some(task_key.to_string())
                    },
                    extra: HashMap::from([("images".into(), 1)]),
                    ..Default::default()
                });
                if let Err(e) = settle_billing(&state.billing, sa, preauth, cost).await {
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

async fn prom_metrics() -> Response {
    let body = tangle_inference_core::metrics::gather();
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(body))
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build metrics response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

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
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1
            }"#,
        )
        .unwrap()
    }
}
