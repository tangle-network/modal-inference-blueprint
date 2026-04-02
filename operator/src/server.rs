//! Axum HTTP server — OpenAI-compatible voice inference proxy.
//!
//! Exposes the same endpoints as Tangle Gateway so developers can hit
//! this operator directly or through the gateway.

use crate::billing::{BillingClient, NonceStore, SpendAuthPayload, UsageUnits};
use crate::config::OperatorConfig;
use crate::idle::IdleManager;
use crate::metrics;
use crate::proxy::ModelRegistry;
use axum::{
    extract::{Json, Multipart, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use alloy::primitives::Address;
use serde::Deserialize;
use blueprint_std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;

// ---------------------------------------------------------------------------
// x402 header constants
// ---------------------------------------------------------------------------

const X402_PAYMENT_SIGNATURE: &str = "X-Payment-Signature";
const X402_PAYMENT_REQUIRED_HEADER: &str = "X-Payment-Required";
const X402_PAYMENT_TOKEN: &str = "X-Payment-Token";
const X402_PAYMENT_RECIPIENT: &str = "X-Payment-Recipient";
const X402_PAYMENT_NETWORK: &str = "X-Payment-Network";

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

pub struct AppState {
    pub registry: ModelRegistry,
    pub config: OperatorConfig,
    pub idle_manager: Option<Arc<IdleManager>>,
    /// Billing client — Some only when billing.required is true.
    pub billing: Option<Arc<BillingClient>>,
    /// Replay-protection nonce store.
    pub nonce_store: Arc<NonceStore>,
    /// This operator's on-chain address (derived from billing.tangle.operator_key).
    pub operator_address: Option<Address>,
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

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SpeechRequest {
    model: Option<String>,
    input: String,
    voice: Option<String>,
    response_format: Option<String>,
    /// SpendAuth for billing (required when billing.required = true).
    /// Can also be provided via X-Payment-Signature header.
    spend_auth: Option<SpendAuthPayload>,
}

// ---------------------------------------------------------------------------
// x402 helpers
// ---------------------------------------------------------------------------

/// Extract SpendAuth from the X-Payment-Signature header.
/// The header value is a JSON SpendAuthPayload, optionally base64 or hex encoded.
fn extract_x402_spend_auth(headers: &HeaderMap) -> Option<SpendAuthPayload> {
    let header_val = headers.get(X402_PAYMENT_SIGNATURE)?.to_str().ok()?;

    // Try direct JSON first
    if let Ok(payload) = serde_json::from_str::<SpendAuthPayload>(header_val) {
        return Some(payload);
    }

    // Try hex-encoded JSON
    let hex_stripped = header_val.strip_prefix("0x").unwrap_or(header_val);
    if let Ok(decoded) = hex::decode(hex_stripped) {
        if let Ok(payload) = serde_json::from_slice::<SpendAuthPayload>(&decoded) {
            return Some(payload);
        }
    }

    None
}

/// Build a 402 Payment Required response with x402 headers.
fn x402_payment_required(state: &AppState, task_type: &str) -> Response {
    let estimated_amount = {
        let units = match task_type {
            "tts" | "clone" | "enhance" => UsageUnits {
                characters: 500,
                ..Default::default()
            },
            "stt" | "diarize" | "translate" | "langid" | "vad" => UsageUnits {
                audio_centiseconds: 3000, // 30 seconds
                ..Default::default()
            },
            "image" | "speakerid" => UsageUnits {
                images: 1,
                ..Default::default()
            },
            _ => UsageUnits {
                prompt_tokens: 1000,
                completion_tokens: 512,
                ..Default::default()
            },
        };
        if let Some(ref billing) = state.billing {
            billing.calculate_cost(task_type, &units)
                .max(state.config.billing.min_charge_amount)
        } else {
            0
        }
    };

    let operator_addr = state
        .operator_address
        .map(|a| format!("{a}"))
        .unwrap_or_default();
    let token_addr = state
        .config
        .billing
        .payment_token_address
        .as_deref()
        .unwrap_or("0x0000000000000000000000000000000000000000");
    let chain_id = state.config.billing.tangle.chain_id.to_string();

    let body = serde_json::json!({
        "error": "payment_required",
        "amount": estimated_amount.to_string(),
        "token": token_addr,
        "recipient": operator_addr,
        "network": chain_id,
        "accepts": ["spend_auth"],
        "description": "ShieldedCredits SpendAuth required. Include spend_auth in request body or X-Payment-Signature header."
    });

    let body_bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = %e, "failed to serialize 402 body");
            return (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };

    Response::builder()
        .status(StatusCode::PAYMENT_REQUIRED)
        .header("content-type", "application/json")
        .header(X402_PAYMENT_REQUIRED_HEADER, estimated_amount.to_string())
        .header(X402_PAYMENT_TOKEN, token_addr)
        .header(X402_PAYMENT_RECIPIENT, &operator_addr)
        .header(X402_PAYMENT_NETWORK, &chain_id)
        .body(axum::body::Body::from(body_bytes))
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "failed to build 402 response");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
        })
}

fn error_json(status: StatusCode, msg: impl Into<String>) -> Response {
    let body = serde_json::json!({ "error": msg.into() });
    (status, axum::Json(body)).into_response()
}

// ---------------------------------------------------------------------------
// Billing validation + settlement
// ---------------------------------------------------------------------------

/// Validate a SpendAuth: amount bounds, operator match, service_id, nonce replay,
/// EIP-712 signature, on-chain account info, min balance.
///
/// Returns the pre-authorized amount on success.
async fn validate_spend_auth(
    state: &AppState,
    spend_auth: &SpendAuthPayload,
) -> Result<u64, Response> {
    let cfg = &state.config.billing;

    // Parse amount
    let requested_amount: u64 = spend_auth.amount.parse().map_err(|_| {
        error_json(
            StatusCode::BAD_REQUEST,
            "invalid spend_auth amount: must be a valid u64 integer",
        )
    })?;

    // Min charge check
    if cfg.min_charge_amount > 0 && requested_amount < cfg.min_charge_amount {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            format!(
                "spend authorization amount ({requested_amount}) is below minimum charge ({})",
                cfg.min_charge_amount
            ),
        ));
    }

    // Max spend check
    if cfg.max_spend_per_request > 0 && requested_amount > cfg.max_spend_per_request {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            format!(
                "spend authorization amount ({requested_amount}) exceeds max_spend_per_request ({})",
                cfg.max_spend_per_request
            ),
        ));
    }

    // Operator address match
    if let Some(op_addr) = state.operator_address {
        let spend_operator: Address = spend_auth.operator.parse().map_err(|_| {
            error_json(StatusCode::BAD_REQUEST, "invalid operator address in spend_auth")
        })?;
        if spend_operator != op_addr {
            return Err(error_json(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend_auth operator ({spend_operator}) does not match this operator ({op_addr})"
                ),
            ));
        }
    }

    // Service ID match
    if let Some(expected_service_id) = cfg.tangle.service_id {
        if spend_auth.service_id != expected_service_id {
            return Err(error_json(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend_auth service_id ({}) does not match operator service ({expected_service_id})",
                    spend_auth.service_id
                ),
            ));
        }
    }

    // Nonce replay protection
    let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
    if state
        .nonce_store
        .check_replay(&nonce_key, cfg.clock_skew_tolerance_secs)
        .await
    {
        return Err(error_json(
            StatusCode::BAD_REQUEST,
            "spend_auth nonce already used (replay detected)",
        ));
    }

    // EIP-712 signature recovery
    let recovered_address = crate::billing::recover_spend_auth_signer(
        spend_auth,
        &cfg.tangle.shielded_credits,
        cfg.tangle.chain_id,
        cfg.clock_skew_tolerance_secs,
    )
    .map_err(|reason| {
        error_json(
            StatusCode::PAYMENT_REQUIRED,
            format!("invalid SpendAuth signature: {reason}"),
        )
    })?;

    // On-chain account verification
    if let Some(ref billing) = state.billing {
        let account_info = billing
            .get_account_info(&spend_auth.commitment)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "failed to check account info");
                error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to verify account info",
                )
            })?;

        if recovered_address != account_info.spending_key {
            return Err(error_json(
                StatusCode::PAYMENT_REQUIRED,
                "SpendAuth signer does not match account spending key",
            ));
        }

        // Min balance check
        if cfg.min_credit_balance > 0
            && account_info.balance < alloy::primitives::U256::from(cfg.min_credit_balance)
        {
            return Err(error_json(
                StatusCode::PAYMENT_REQUIRED,
                format!(
                    "credit balance ({}) is below minimum required ({})",
                    account_info.balance, cfg.min_credit_balance
                ),
            ));
        }
    }

    Ok(requested_amount)
}

/// Pre-authorize on-chain and mark nonce as used.
async fn authorize_billing(
    state: &AppState,
    spend_auth: &SpendAuthPayload,
    preauth_amount: u64,
) -> Result<(), Response> {
    let Some(ref billing) = state.billing else {
        return Ok(());
    };

    if let Err(e) = billing.authorize_spend(spend_auth).await {
        tracing::error!(error = %e, "authorizeSpend failed");
        return Err(error_json(
            StatusCode::PAYMENT_REQUIRED,
            format!("billing authorization failed: {e}"),
        ));
    }

    let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
    state
        .nonce_store
        .insert(
            nonce_key,
            spend_auth.expiry,
            state.config.billing.clock_skew_tolerance_secs,
        )
        .await;

    tracing::info!(preauth_amount, "billing pre-authorized");
    Ok(())
}

/// Settle billing post-response: calculate actual cost, claim payment on-chain.
async fn settle_billing(
    state: &AppState,
    spend_auth: &SpendAuthPayload,
    preauth_amount: u64,
    task_type: &str,
    usage: &UsageUnits,
) {
    let Some(ref billing) = state.billing else {
        return;
    };

    let actual_cost = billing.calculate_cost(task_type, usage);
    let charge_amount = actual_cost.min(preauth_amount);

    tracing::info!(
        actual_cost,
        preauth_amount,
        charge_amount,
        task_type,
        "settling billing (contract settles full pre-auth)"
    );

    if charge_amount > 0 {
        if let Err(e) = billing.claim_payment(spend_auth, charge_amount).await {
            tracing::error!(
                error = %e,
                charge_amount,
                "billing settlement failed — revenue lost"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Wake idle model helper
// ---------------------------------------------------------------------------

async fn ensure_model_awake(state: &AppState, model_name: &str) -> Result<(), Response> {
    if let Some(ref mgr) = state.idle_manager {
        if !mgr.record_request(model_name).await {
            if let Err(e) = mgr.wake_model(model_name).await {
                return Err(
                    (StatusCode::SERVICE_UNAVAILABLE, format!("Model waking: {e}")).into_response()
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /v1/audio/speech — OpenAI-compatible TTS
async fn synthesize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<SpeechRequest>,
) -> Response {
    let model_name = body.model.as_deref().unwrap_or("default").to_string();

    let model = state
        .registry
        .get(&model_name)
        .or_else(|| state.registry.list_by_type("tts").into_iter().next());

    let Some(model) = model else {
        return (StatusCode::BAD_REQUEST, "No TTS model configured").into_response();
    };

    // x402: pull SpendAuth from header if not in body
    if body.spend_auth.is_none() {
        body.spend_auth = extract_x402_spend_auth(&headers);
    }

    // Billing gate
    let preauth_amount = if state.config.billing.required {
        let Some(ref spend_auth) = body.spend_auth else {
            return x402_payment_required(&state, &model.task_type);
        };
        let amount = match validate_spend_auth(&state, spend_auth).await {
            Ok(a) => a,
            Err(r) => return r,
        };
        if let Err(r) = authorize_billing(&state, spend_auth, amount).await {
            return r;
        }
        Some(amount)
    } else {
        None
    };

    if let Err(r) = ensure_model_awake(&state, &model.name).await {
        return r;
    }

    let chars = body.input.len() as u64;

    let payload = serde_json::json!({
        "text": body.input,
        "voice_id": body.voice.as_deref().unwrap_or("default"),
        "format": body.response_format.as_deref().unwrap_or("wav"),
    });

    match state
        .registry
        .proxy_request(
            &model.name,
            None,
            Bytes::from(serde_json::to_vec(&payload).unwrap()),
            "application/json",
        )
        .await
    {
        Ok(resp) => {
            metrics::CHARACTERS_TOTAL
                .with_label_values(&[&model.name])
                .inc_by(chars);

            // Settle billing post-response
            if let (Some(ref spend_auth), Some(preauth)) = (&body.spend_auth, preauth_amount) {
                let usage = UsageUnits {
                    characters: chars,
                    ..Default::default()
                };
                settle_billing(&state, spend_auth, preauth, &model.task_type, &usage).await;
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
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Proxy error: {e}")).into_response(),
    }
}

/// POST /v1/audio/transcriptions — OpenAI-compatible STT
async fn transcribe(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let mut audio_data: Option<Bytes> = None;
    let mut model_name = "default".to_string();
    let mut spend_auth_json: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("file") => {
                audio_data = field.bytes().await.ok();
            }
            Some("model") => {
                model_name = field.text().await.unwrap_or_default();
            }
            Some("spend_auth") => {
                spend_auth_json = field.text().await.ok();
            }
            _ => {}
        }
    }

    let Some(audio) = audio_data else {
        return (StatusCode::BAD_REQUEST, "Missing audio file").into_response();
    };

    let model = state
        .registry
        .get(&model_name)
        .or_else(|| state.registry.list_by_type("stt").into_iter().next());

    let Some(model) = model else {
        return (StatusCode::BAD_REQUEST, "No STT model configured").into_response();
    };

    // Resolve SpendAuth: multipart field > header
    let spend_auth: Option<SpendAuthPayload> = spend_auth_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .or_else(|| extract_x402_spend_auth(&headers));

    // Billing gate
    let preauth_amount = if state.config.billing.required {
        let Some(ref sa) = spend_auth else {
            return x402_payment_required(&state, &model.task_type);
        };
        let amount = match validate_spend_auth(&state, sa).await {
            Ok(a) => a,
            Err(r) => return r,
        };
        if let Err(r) = authorize_billing(&state, sa, amount).await {
            return r;
        }
        Some(amount)
    } else {
        None
    };

    if let Err(r) = ensure_model_awake(&state, &model.name).await {
        return r;
    }

    let audio_len = audio.len();

    match state
        .registry
        .proxy_request(&model.name, None, audio, "audio/wav")
        .await
    {
        Ok(resp) => {
            // Settle billing: estimate audio duration from file size (rough heuristic:
            // 16kHz mono 16-bit PCM = 32000 bytes/sec)
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let centiseconds = (audio_len as u64 * 100) / 32000;
                let usage = UsageUnits {
                    audio_centiseconds: centiseconds,
                    ..Default::default()
                };
                settle_billing(&state, sa, preauth, &model.task_type, &usage).await;
            }

            (
                StatusCode::OK,
                [("content-type", "application/json")],
                resp.data,
            )
                .into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Proxy error: {e}")).into_response(),
    }
}

/// POST /proxy/:model/*path — raw proxy to any model endpoint
async fn proxy_raw(
    State(state): State<Arc<AppState>>,
    Path((model, path)): Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let model_cfg = state.registry.get(&model);
    let task_type = model_cfg
        .map(|m| m.task_type.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Billing gate for raw proxy: SpendAuth must be in X-Payment-Signature header
    let spend_auth = extract_x402_spend_auth(&headers);
    let preauth_amount = if state.config.billing.required {
        let Some(ref sa) = spend_auth else {
            return x402_payment_required(&state, &task_type);
        };
        let amount = match validate_spend_auth(&state, sa).await {
            Ok(a) => a,
            Err(r) => return r,
        };
        if let Err(r) = authorize_billing(&state, sa, amount).await {
            return r;
        }
        Some(amount)
    } else {
        None
    };

    if let Err(r) = ensure_model_awake(&state, &model).await {
        return r;
    }

    let path = format!("/{path}");
    match state
        .registry
        .proxy_request(&model, Some(&path), body, "application/json")
        .await
    {
        Ok(resp) => {
            // Settle with a single-unit default for raw proxy (operator can refine)
            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let usage = UsageUnits {
                    images: 1,
                    ..Default::default()
                };
                settle_billing(&state, sa, preauth, &task_type, &usage).await;
            }
            (
                StatusCode::OK,
                [("content-type", resp.content_type.as_str())],
                resp.data,
            )
                .into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Proxy error: {e}")).into_response(),
    }
}

/// GET /v1/audio/models — list available models
async fn list_models(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let models: Vec<serde_json::Value> = state
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

    axum::Json(serde_json::json!({ "object": "list", "data": models }))
}

/// GET /health — full service instance metadata + aggregated metrics
async fn health(State(state): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
    let model_health = state.registry.health_check_all().await;
    let all_ok = model_health.iter().all(|h| h.status == "ok");

    let models: Vec<serde_json::Value> = model_health
        .iter()
        .map(|h| {
            let model_config = state.registry.get(&h.name);
            serde_json::json!({
                "name": h.name,
                "status": h.status,
                "type": model_config.map(|m| m.task_type.as_str()).unwrap_or("unknown"),
                "latency_ms": h.latency_ms,
                "modal_endpoint": model_config.map(|m| m.modal_endpoint.as_str()),
                "modal_app_name": model_config.map(|m| {
                    let url = m.modal_endpoint.as_str();
                    url.find("--").map(|i| &url[i+2..url.find(".modal").unwrap_or(url.len())]).unwrap_or("")
                }),
            })
        })
        .collect();

    axum::Json(serde_json::json!({
        "status": if all_ok { "ok" } else { "degraded" },
        "operator": state.config.name,
        "billing_required": state.config.billing.required,
        "models": models,
        "metrics": metrics::health_summary(),
    }))
}

/// GET /metrics — Prometheus text format
async fn prom_metrics() -> String {
    metrics::gather()
}
