//! Auto-registration with the ph0ny marketplace.
//!
//! On startup, the operator registers its models with the ph0ny Gateway
//! so traffic can be routed to it.

use crate::config::OperatorConfig;
use reqwest::Client;
use tracing::{info, warn, error};

/// Register this operator's models with the ph0ny marketplace.
pub async fn register_with_gateway(config: &OperatorConfig) -> anyhow::Result<()> {
    let client = Client::new();
    let gateway_url = &config.gateway.url;

    for model in &config.models {
        let payload = serde_json::json!({
            "name": format!("{} — {}", config.name, model.name),
            "type": model.task_type,
            "endpointUrl": model.modal_endpoint,
            "description": format!("Operated by {}", config.name),
            "pricingModel": match model.task_type.as_str() {
                "tts" | "clone" | "enhance" => "per-character",
                "stt" | "diarize" | "translate" | "vad" => "per-minute",
                _ => "per-character",
            },
            "payoutEmail": config.gateway.payout_email,
        });

        // Try to register — may fail if gateway is unreachable or model already registered
        match client
            .post(format!("{gateway_url}/marketplace/providers/register"))
            .json(&payload)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                info!(model = %model.name, "Registered with ph0ny marketplace");
            }
            Ok(resp) if resp.status().as_u16() == 409 => {
                info!(model = %model.name, "Already registered with ph0ny marketplace");
            }
            Ok(resp) => {
                warn!(
                    model = %model.name,
                    status = %resp.status(),
                    "Failed to register with marketplace (will retry on next restart)"
                );
            }
            Err(e) => {
                warn!(
                    model = %model.name,
                    error = %e,
                    "Could not reach ph0ny marketplace (operating standalone)"
                );
            }
        }
    }

    Ok(())
}
