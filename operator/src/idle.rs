//! Intelligent Modal app lifecycle management.
//!
//! Tracks per-model last-request timestamps. A background task periodically
//! checks for idle models and stops their Modal apps to save GPU cost.
//! On next request, the proxy detects the stopped state and wakes the app.

use blueprint_std::collections::HashMap;
use blueprint_std::sync::Arc;
use blueprint_std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Per-model idle state.
#[derive(Debug)]
struct ModelState {
    last_request: Instant,
    stopped: bool,
    modal_app_name: Option<String>,
}

/// Tracks idle state for all models and manages Modal app lifecycle.
pub struct IdleManager {
    states: RwLock<HashMap<String, ModelState>>,
    idle_threshold: Duration,
    check_interval: Duration,
}

impl IdleManager {
    pub fn new(idle_threshold_mins: u64, check_interval_mins: u64) -> Arc<Self> {
        Arc::new(Self {
            states: RwLock::new(HashMap::new()),
            idle_threshold: Duration::from_secs(idle_threshold_mins * 60),
            check_interval: Duration::from_secs(check_interval_mins * 60),
        })
    }

    /// Register a model with its Modal app name (extracted from endpoint URL).
    pub async fn register_model(&self, model_name: &str, modal_endpoint: &str) {
        // Extract app name from Modal URL: https://org--app-name.modal.run → app-name
        let app_name = extract_modal_app_name(modal_endpoint);
        let mut states = self.states.write().await;
        states.insert(model_name.to_string(), ModelState {
            last_request: Instant::now(),
            stopped: false,
            modal_app_name: app_name,
        });
    }

    /// Record a request for a model. Returns true if model is available,
    /// false if it's stopped and needs waking.
    pub async fn record_request(&self, model_name: &str) -> bool {
        let mut states = self.states.write().await;
        if let Some(state) = states.get_mut(model_name) {
            state.last_request = Instant::now();
            if state.stopped {
                // Model was stopped — caller should wake it
                return false;
            }
        }
        true
    }

    /// Mark a model as running (after wake completes).
    pub async fn mark_running(&self, model_name: &str) {
        let mut states = self.states.write().await;
        if let Some(state) = states.get_mut(model_name) {
            state.stopped = false;
        }
    }

    /// Check if a model is currently stopped.
    pub async fn is_stopped(&self, model_name: &str) -> bool {
        let states = self.states.read().await;
        states.get(model_name).map(|s| s.stopped).unwrap_or(false)
    }

    /// Wake a stopped Modal app. Blocks until ready or timeout.
    pub async fn wake_model(&self, model_name: &str) -> anyhow::Result<()> {
        let app_name = {
            let states = self.states.read().await;
            states.get(model_name)
                .and_then(|s| s.modal_app_name.clone())
        };

        let Some(app_name) = app_name else {
            warn!(model = model_name, "No Modal app name — cannot wake");
            return Ok(());
        };

        info!(model = model_name, app = %app_name, "Waking Modal app");

        let output = Command::new("modal")
            .args(["app", "deploy", &app_name])
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!(model = model_name, stderr = %stderr, "Failed to wake Modal app");
            return Err(anyhow::anyhow!("modal app deploy failed: {stderr}"));
        }

        self.mark_running(model_name).await;
        info!(model = model_name, "Modal app woken");
        Ok(())
    }

    /// Background task: check for idle models and stop them.
    pub async fn run_idle_checker(self: Arc<Self>) {
        let mut interval = tokio::time::interval(self.check_interval);
        loop {
            interval.tick().await;
            self.check_and_stop_idle().await;
        }
    }

    async fn check_and_stop_idle(&self) {
        let now = Instant::now();
        let mut to_stop: Vec<(String, String)> = Vec::new();

        {
            let states = self.states.read().await;
            for (name, state) in states.iter() {
                if state.stopped {
                    continue;
                }
                if now.duration_since(state.last_request) > self.idle_threshold {
                    if let Some(ref app) = state.modal_app_name {
                        to_stop.push((name.clone(), app.clone()));
                    }
                }
            }
        }

        for (model_name, app_name) in to_stop {
            info!(
                model = %model_name,
                app = %app_name,
                idle_mins = self.idle_threshold.as_secs() / 60,
                "Stopping idle Modal app"
            );

            match Command::new("modal")
                .args(["app", "stop", &app_name])
                .output()
                .await
            {
                Ok(output) if output.status.success() => {
                    let mut states = self.states.write().await;
                    if let Some(state) = states.get_mut(&model_name) {
                        state.stopped = true;
                    }
                    info!(model = %model_name, "Modal app stopped");
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    warn!(model = %model_name, stderr = %stderr, "Failed to stop Modal app");
                }
                Err(e) => {
                    warn!(model = %model_name, err = %e, "modal CLI not available");
                }
            }
        }
    }
}

/// Extract Modal app name from endpoint URL.
/// "https://drewstone--cosyvoice3-service.modal.run" → "cosyvoice3-service"
fn extract_modal_app_name(url: &str) -> Option<String> {
    let url = url.trim_end_matches('/');
    // Pattern: https://{org}--{app-name}.modal.run
    if let Some(host_start) = url.find("//") {
        let rest = &url[host_start + 2..];
        if let Some(dot) = rest.find(".modal.run") {
            let host = &rest[..dot];
            if let Some(sep) = host.find("--") {
                return Some(host[sep + 2..].to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_modal_app_name() {
        assert_eq!(
            extract_modal_app_name("https://drewstone--cosyvoice3-service.modal.run"),
            Some("cosyvoice3-service".to_string())
        );
        assert_eq!(
            extract_modal_app_name("https://org--my-app.modal.run/"),
            Some("my-app".to_string())
        );
        assert_eq!(
            extract_modal_app_name("http://localhost:8000"),
            None
        );
    }
}
