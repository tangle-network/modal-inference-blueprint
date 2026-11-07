//! Axum HTTP server — OpenAI-compatible proxy to Modal deployments.
//!
//! All shared infrastructure (nonce store, spend-auth validation, x402 headers,
//! metrics, app state container) lives in `tangle-inference-core`. This module
//! contains the modal-specific backend struct, OpenAI-compatible request
//! handlers, and task-type aware cost calculation via `TaskTypeCostModel`.
use crate::config::{ModelEndpoint, OperatorConfig};
use crate::proxy::ModelRegistry;
use axum::{
    extract::{Json, Multipart, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tangle_inference_core::server::{
    authorize_request, error_response, metrics_handler, payment_required, resolve_payment_proof,
    settle_request, AuthorizedPayment,
};
use tangle_inference_core::{
    AppState, CostModel, CostParams, FlatRequestCostModel, PerCharCostModel, PerImageCostModel,
    PerSecondCostModel, PerTokenCostModel, SpendAuthPayload, TaskTypeCostModel,
};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
/// Backend state for the Modal proxy. Holds the model registry,
/// pre-built task-type cost model, and a reference to the full operator config
/// (for modal-specific knobs only — all shared knobs live on `AppState`).
pub struct ModalBackend {
    pub config: Arc<OperatorConfig>,
    pub registry: ModelRegistry,
    pub cost_model: Arc<TaskTypeCostModel>,
    usage: UsageLedger,
}
struct UsageLedger {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}
impl UsageLedger {
    const fn new() -> Self {
        Self {
            requests: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
        }
    }
    fn record(&self, prompt_tokens: u64, completion_tokens: u64) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);
    }
}
impl ModalBackend {
    pub fn new(config: Arc<OperatorConfig>, registry: ModelRegistry) -> Self {
        let modal = &config.modal;
        let per_task: HashMap<String, Box<dyn CostModel>> = [
            (
                "chat",
                token_cost(modal.price_per_input_token, modal.price_per_output_token),
            ),
            ("tts", char_cost(modal.price_per_1k_tts_chars)),
            ("stt", second_cost(modal.price_per_stt_second)),
            ("image", image_cost(modal.price_per_image)),
            ("video", second_cost(modal.price_per_video_second)),
            ("music", second_cost(modal.price_per_music_second)),
            (
                "embedding",
                token_cost(modal.price_per_1k_embedding_tokens, 0),
            ),
        ]
        .into_iter()
        .map(|(key, model)| (key.to_string(), model))
        .collect();
        let cost_model = Arc::new(TaskTypeCostModel {
            default: Box::new(FlatRequestCostModel {
                price_per_request: modal.default_price_per_request,
            }),
            per_task,
        });
        Self {
            config,
            registry,
            cost_model,
            usage: UsageLedger::new(),
        }
    }
    pub fn record_usage(&self, prompt_tokens: u64, completion_tokens: u64) {
        self.usage.record(prompt_tokens, completion_tokens);
    }
    pub fn usage_snapshot(&self) -> (u64, u64, u64) {
        (
            self.usage.requests.load(Ordering::Relaxed),
            self.usage.prompt_tokens.load(Ordering::Relaxed),
            self.usage.completion_tokens.load(Ordering::Relaxed),
        )
    }
    /// Calculate cost for a given task type and usage.
    pub fn calculate_cost(&self, params: &CostParams) -> u64 {
        self.cost_model.calculate_cost(params)
    }
}
fn token_cost(input: u64, output: u64) -> Box<dyn CostModel> {
    Box::new(PerTokenCostModel {
        price_per_input_token: input,
        price_per_output_token: output,
    })
}
fn char_cost(price: u64) -> Box<dyn CostModel> {
    Box::new(PerCharCostModel {
        price_per_1k_chars: price,
    })
}
fn second_cost(price: u64) -> Box<dyn CostModel> {
    Box::new(PerSecondCostModel {
        price_per_second: price,
    })
}
fn image_cost(price: u64) -> Box<dyn CostModel> {
    Box::new(PerImageCostModel {
        price_per_image: price,
    })
}
pub(crate) fn canonical_task_key(task_type: &str) -> Option<&'static str> {
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
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/audio/speech", post(synthesize))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/usage", get(usage_report))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/videos/generations", post(videos_generations))
        .route("/v1/embeddings", post(embeddings))
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/models", get(list_models))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            tangle_inference_core::server::authenticate_endpoint,
        ))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TimeoutLayer::new(Duration::from_secs(120)))
}
/// Start the HTTP server with graceful shutdown, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    state
        .backend::<ModalBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not a ModalBackend"))?;
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
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
fn resolve_model<'a>(
    backend: &'a ModalBackend,
    name: Option<&str>,
    fallback_task: &str,
) -> Option<&'a ModelEndpoint> {
    name.and_then(|n| backend.registry.get(n)).or_else(|| {
        backend
            .registry
            .list_by_type(fallback_task)
            .into_iter()
            .next()
    })
}
/// Authorizes the request's payment; None when billing is optional and the caller sent no proof.
#[expect(
    clippy::result_large_err,
    reason = "HTTP handlers return this error response directly without an extra allocation."
)]
async fn authorize(
    state: &AppState,
    headers: &HeaderMap,
    spend_auth: Option<SpendAuthPayload>,
    cost: u64,
) -> Result<Option<AuthorizedPayment>, Response> {
    let Some(proof) = resolve_payment_proof(headers, spend_auth, None) else {
        let b = &state.billing_config;
        let floor = cost.max(b.min_charge_amount);
        let refuse = || payment_required(b, &state.tangle_config, state.operator_address, floor);
        return (!b.billing_required).then_some(None).ok_or_else(refuse);
    };
    // Probing would wake a scale-to-zero GPU app; the proxy reports upstream errors instead.
    let ready = std::future::ready(true);
    authorize_request(state, proof, cost, ready).await.map(Some)
}
/// Wake the model, proxy its request, and settle the authorized cost; a failure releases it.
async fn proxy_and_settle(
    state: &AppState,
    backend: &ModalBackend,
    model_name: &str,
    payload: Bytes,
    content_type: &str,
    payment: Option<AuthorizedPayment>,
    actual_cost: u64,
) -> Response {
    match backend
        .registry
        .proxy_request(model_name, payload, content_type)
        .await
    {
        Ok(resp) => {
            let (prompt_tokens, completion_tokens) = usage_from_body(&resp.data);
            backend.record_usage(prompt_tokens, completion_tokens);
            if let Some(auth) = &payment {
                settle_request(state, auth, Some(actual_cost)).await;
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
fn usage_from_body(data: &[u8]) -> (u64, u64) {
    serde_json::from_slice::<serde_json::Value>(data)
        .ok()
        .and_then(|value| {
            let usage = value.get("usage")?;
            Some((
                usage
                    .get("prompt_tokens")
                    .and_then(|token| token.as_u64())
                    .unwrap_or(0),
                usage
                    .get("completion_tokens")
                    .and_then(|token| token.as_u64())
                    .unwrap_or(0),
            ))
        })
        .unwrap_or((0, 0))
}
async fn usage_report(State(state): State<AppState>) -> impl IntoResponse {
    let backend = backend_from(&state);
    let (requests, prompt_tokens, completion_tokens) = backend.usage_snapshot();
    Json(serde_json::json!({
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }))
}
#[expect(
    clippy::result_large_err,
    reason = "HTTP handlers return this error response directly without an extra allocation."
)]
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
    let cost_params = CostParams {
        task_type: Some("tts".into()),
        extra: HashMap::from([("characters".into(), chars)]),
        ..Default::default()
    };
    let estimated_cost = backend.calculate_cost(&CostParams {
        extra: HashMap::from([("characters".into(), chars.max(500))]),
        ..cost_params.clone()
    });
    let payment = match authorize(&state, &headers, body.spend_auth, estimated_cost).await {
        Ok(payment) => payment,
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
        &state,
        backend,
        &model.name,
        payload_bytes,
        "application/json",
        payment,
        backend.calculate_cost(&cost_params),
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
    let spend_auth = spend_auth_json.and_then(|s| serde_json::from_str(&s).ok());
    let payment = match authorize(&state, &headers, spend_auth, estimated_cost).await {
        Ok(payment) => payment,
        Err(resp) => return resp,
    };
    proxy_and_settle(
        &state,
        backend,
        &model.name,
        audio,
        "audio/wav",
        payment,
        backend.calculate_cost(&cost_params),
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
        return error_response(
            StatusCode::BAD_REQUEST,
            "No chat model configured".into(),
            "invalid_request_error",
            "no_model",
        );
    };
    let est_prompt: u32 = body
        .messages
        .iter()
        .map(|m| (m.content.len() as u32) / 4 + 1)
        .sum();
    let estimated_cost = backend.calculate_cost(&CostParams {
        task_type: Some("chat".into()),
        prompt_tokens: est_prompt,
        completion_tokens: body.max_tokens,
        ..Default::default()
    });
    let payment = match authorize(&state, &headers, body.spend_auth, estimated_cost).await {
        Ok(payment) => payment,
        Err(resp) => return resp,
    };
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
    match backend
        .registry
        .proxy_request(&model.name, payload_bytes, "application/json")
        .await
    {
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
            backend.record_usage(u64::from(pt), u64::from(ct));
            if let Some(auth) = &payment {
                let cost = backend.calculate_cost(&CostParams {
                    task_type: Some("chat".into()),
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..Default::default()
                });
                settle_request(&state, auth, Some(cost)).await;
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
    let cost = backend.calculate_cost(&CostParams {
        task_type: Some("image".into()),
        extra: HashMap::from([("images".into(), body.n.max(1) as u64)]),
        ..Default::default()
    });
    let payment = match authorize(&state, &headers, body.spend_auth, cost).await {
        Ok(payment) => payment,
        Err(resp) => return resp,
    };
    let payload = serde_json::json!({"model": model.name, "prompt": body.prompt, "n": body.n});
    let payload_bytes = match json_payload(&payload) {
        Ok(b) => b,
        Err(r) => return r,
    };
    proxy_and_settle(
        &state,
        backend,
        &model.name,
        payload_bytes,
        "application/json",
        payment,
        cost,
    )
    .await
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
    let cost = backend.calculate_cost(&CostParams {
        task_type: Some("video".into()),
        extra: HashMap::from([("centiseconds".into(), (body.duration_seconds as u64) * 100)]),
        ..Default::default()
    });
    let payment = match authorize(&state, &headers, body.spend_auth, cost).await {
        Ok(payment) => payment,
        Err(resp) => return resp,
    };
    let payload = serde_json::json!({"model": model.name, "prompt": body.prompt, "duration_seconds": body.duration_seconds});
    let payload_bytes = match json_payload(&payload) {
        Ok(b) => b,
        Err(r) => return r,
    };
    proxy_and_settle(
        &state,
        backend,
        &model.name,
        payload_bytes,
        "application/json",
        payment,
        cost,
    )
    .await
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
    let approx_chars: u64 = match &body.input {
        serde_json::Value::String(s) => s.len() as u64,
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.len() as u64))
            .sum(),
        _ => 0,
    };
    let cost = backend.calculate_cost(&CostParams {
        task_type: Some("embedding".into()),
        prompt_tokens: (approx_chars / 4).max(1) as u32,
        ..Default::default()
    });
    let payment = match authorize(&state, &headers, body.spend_auth, cost).await {
        Ok(payment) => payment,
        Err(resp) => return resp,
    };
    let payload = serde_json::json!({"model": model.name, "input": body.input});
    let payload_bytes = match json_payload(&payload) {
        Ok(b) => b,
        Err(r) => return r,
    };
    proxy_and_settle(
        &state,
        backend,
        &model.name,
        payload_bytes,
        "application/json",
        payment,
        cost,
    )
    .await
}
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
                "object": "model",
            })
        })
        .collect();
    Json(serde_json::json!({ "object": "list", "data": models }))
}
/// Reports operator state without probing upstreams: each probe would wake a scale-to-zero GPU app.
async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    Json(serde_json::json!({
        "status": "ok",
        "operator": backend.config.name,
        "billing_required": state.billing_config.billing_required,
        "models": backend.registry.list().iter().map(|m| serde_json::json!({"name": m.name, "type": m.task_type})).collect::<Vec<_>>(),
        "metrics": tangle_inference_core::metrics::health_summary(),
    }))
}
