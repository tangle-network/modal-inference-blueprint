//! Axum HTTP server — OpenAI-compatible voice inference proxy.
//!
//! Exposes the same endpoints as ph0ny Gateway so developers can hit
//! this operator directly or through the gateway.

use crate::config::OperatorConfig;
use crate::idle::IdleManager;
use crate::metrics;
use crate::proxy::ModelRegistry;
use axum::{
    extract::{Json, Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tracing::info;

pub struct AppState {
    pub registry: ModelRegistry,
    pub config: OperatorConfig,
    pub idle_manager: Option<Arc<IdleManager>>,
}

/// Build the Axum router.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        // OpenAI-compatible
        .route("/v1/audio/speech", post(synthesize))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/models", get(list_models))
        // Direct proxy (any model, any path)
        .route("/proxy/:model/*path", post(proxy_raw))
        // Health + metrics
        .route("/health", get(health))
        .route("/metrics", get(prom_metrics))
        .route("/models", get(list_models))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(120)))
}

#[derive(Deserialize)]
struct SpeechRequest {
    model: Option<String>,
    input: String,
    voice: Option<String>,
    response_format: Option<String>,
}

/// POST /v1/audio/speech — OpenAI-compatible TTS
async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SpeechRequest>,
) -> Response {
    let model_name = body.model.as_deref().unwrap_or("default");

    // Find a TTS model
    let model = state.registry.get(model_name)
        .or_else(|| state.registry.list_by_type("tts").into_iter().next());

    let Some(model) = model else {
        return (StatusCode::BAD_REQUEST, "No TTS model configured").into_response();
    };

    // Idle tracking: record request, wake if stopped
    if let Some(ref mgr) = state.idle_manager {
        if !mgr.record_request(&model.name).await {
            if let Err(e) = mgr.wake_model(&model.name).await {
                return (StatusCode::SERVICE_UNAVAILABLE, format!("Model waking: {e}")).into_response();
            }
        }
    }

    let payload = serde_json::json!({
        "text": body.input,
        "voice_id": body.voice.as_deref().unwrap_or("default"),
        "format": body.response_format.as_deref().unwrap_or("wav"),
    });

    match state.registry.proxy_request(
        &model.name,
        None,
        Bytes::from(serde_json::to_vec(&payload).unwrap()),
        "application/json",
    ).await {
        Ok(resp) => {
            let chars = body.input.len() as u64;
            metrics::CHARACTERS_TOTAL.with_label_values(&[&model.name]).inc_by(chars);

            (
                StatusCode::OK,
                [
                    ("content-type", resp.content_type.as_str()),
                    ("x-model", &model.name),
                    ("x-latency-ms", &resp.latency_ms.to_string()),
                ],
                resp.data,
            ).into_response()
        }
        Err(e) => {
            (StatusCode::BAD_GATEWAY, format!("Proxy error: {e}")).into_response()
        }
    }
}

/// POST /v1/audio/transcriptions — OpenAI-compatible STT
async fn transcribe(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut audio_data: Option<Bytes> = None;
    let mut model_name = "default".to_string();

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("file") => {
                audio_data = field.bytes().await.ok();
            }
            Some("model") => {
                model_name = field.text().await.unwrap_or_default();
            }
            _ => {}
        }
    }

    let Some(audio) = audio_data else {
        return (StatusCode::BAD_REQUEST, "Missing audio file").into_response();
    };

    let model = state.registry.get(&model_name)
        .or_else(|| state.registry.list_by_type("stt").into_iter().next());

    let Some(model) = model else {
        return (StatusCode::BAD_REQUEST, "No STT model configured").into_response();
    };

    match state.registry.proxy_request(
        &model.name,
        None,
        audio,
        "audio/wav",
    ).await {
        Ok(resp) => {
            (StatusCode::OK, [("content-type", "application/json")], resp.data).into_response()
        }
        Err(e) => {
            (StatusCode::BAD_GATEWAY, format!("Proxy error: {e}")).into_response()
        }
    }
}

/// POST /proxy/:model/*path — raw proxy to any model endpoint
async fn proxy_raw(
    State(state): State<Arc<AppState>>,
    Path((model, path)): Path<(String, String)>,
    body: Bytes,
) -> Response {
    let path = format!("/{path}");
    match state.registry.proxy_request(&model, Some(&path), body, "application/json").await {
        Ok(resp) => {
            (StatusCode::OK, [("content-type", resp.content_type.as_str())], resp.data).into_response()
        }
        Err(e) => {
            (StatusCode::BAD_GATEWAY, format!("Proxy error: {e}")).into_response()
        }
    }
}

/// GET /v1/audio/models — list available models
async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let models: Vec<serde_json::Value> = state.registry.list().iter().map(|m| {
        serde_json::json!({
            "id": m.name,
            "type": m.task_type,
            "endpoint": m.modal_endpoint,
            "object": "model",
        })
    }).collect();

    Json(serde_json::json!({ "object": "list", "data": models }))
}

/// GET /health
async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let model_health = state.registry.health_check_all().await;
    let all_ok = model_health.iter().all(|h| h.status == "ok");

    Json(serde_json::json!({
        "status": if all_ok { "ok" } else { "degraded" },
        "operator": state.config.name,
        "models": model_health.iter().map(|h| serde_json::json!({
            "name": h.name,
            "status": h.status,
            "latency_ms": h.latency_ms,
        })).collect::<Vec<_>>(),
    }))
}

/// GET /metrics — Prometheus text format
async fn prom_metrics() -> String {
    metrics::gather()
}
