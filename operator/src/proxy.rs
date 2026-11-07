//! Modal proxy — routes requests to operator's Modal deployments.
//!
//! This is the core of the blueprint: a secure, metered proxy that sits between
//! the Tangle Gateway and the operator's Modal-hosted models.

use crate::config::ModelEndpoint;
use anyhow::Result;
use bytes::Bytes;
use reqwest::Client;
use std::collections::HashMap;
use std::time::Instant;
use tangle_inference_core::RequestGuard;
use tracing::{error, info};

/// Model registry — maps model names to their Modal endpoints.
pub struct ModelRegistry {
    models: HashMap<String, ModelEndpoint>,
    client: Client,
}

impl ModelRegistry {
    pub fn new(models: Vec<ModelEndpoint>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("HTTP client");

        let map: HashMap<String, ModelEndpoint> =
            models.into_iter().map(|m| (m.name.clone(), m)).collect();

        info!(models = map.len(), "Model registry initialized");
        for (name, m) in &map {
            info!(name, endpoint = %m.modal_endpoint, task = %m.task_type, "Registered model");
        }

        Self {
            models: map,
            client,
        }
    }

    /// Get a model endpoint by name.
    pub fn get(&self, name: &str) -> Option<&ModelEndpoint> {
        self.models.get(name)
    }

    /// List all registered models.
    pub fn list(&self) -> Vec<&ModelEndpoint> {
        self.models.values().collect()
    }

    /// List models by task type.
    pub fn list_by_type(&self, task_type: &str) -> Vec<&ModelEndpoint> {
        self.models
            .values()
            .filter(|m| m.task_type == task_type)
            .collect()
    }

    /// Proxy a request to a Modal endpoint.
    pub async fn proxy_request(
        &self,
        model_name: &str,
        body: Bytes,
        content_type: &str,
    ) -> Result<ProxyResponse> {
        let model = self
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {model_name}"))?;

        let inference_path = model.resolve_inference_path();
        let url = format!(
            "{}{}",
            model.modal_endpoint.trim_end_matches('/'),
            inference_path
        );

        let mut guard = RequestGuard::new(model_name);
        let start = Instant::now();

        let response = self
            .client
            .post(&url)
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await?;

        let status = response.status();
        let latency_ms = start.elapsed().as_millis() as u64;

        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(model = model_name, status = %status, latency_ms, "Modal proxy error");
            guard.record_error("upstream");
            return Err(anyhow::anyhow!("Modal returned {status}: {error_body}"));
        }

        let response_content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();

        let data = response.bytes().await?;

        info!(
            model = model_name,
            latency_ms,
            bytes = data.len(),
            "Modal proxy success"
        );
        guard.set_success();
        drop(guard);

        Ok(ProxyResponse {
            data,
            content_type: response_content_type,
            latency_ms,
        })
    }
}

pub struct ProxyResponse {
    pub data: Bytes,
    pub content_type: String,
    pub latency_ms: u64,
}
