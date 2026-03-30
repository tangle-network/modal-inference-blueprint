//! Auto-registration with the Tangle marketplace.
//!
//! On startup, the operator registers its models with the Tangle Gateway
//! so traffic can be routed to it.

use crate::config::OperatorConfig;
use reqwest::Client;
use tracing::{info, warn, error};

/// Register this operator's models with the Tangle marketplace.
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
                "tts" | "clone" | "voice-design" => "per-character",
                "stt" | "diarize" | "translate" | "vad" | "s2s" | "speakerid" | "langid" => "per-second-audio",
                "video-generation" | "video-avatar" | "video-lipsync" | "video-understanding" => "per-second-video",
                "image-generation" => "per-image",
                "music-generation" => "per-second-music",
                "video-stitch" | "audio-processing" | "voice-conversion" | "enhance" => "per-job",
                "text-generation" | "embedding" | "rerank" => "per-million-tokens",
                _ => "per-job",
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
                info!(model = %model.name, "Registered with Tangle marketplace");
            }
            Ok(resp) if resp.status().as_u16() == 409 => {
                info!(model = %model.name, "Already registered with Tangle marketplace");
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
                    "Could not reach Tangle marketplace (operating standalone)"
                );
            }
        }
    }

    Ok(())
}
