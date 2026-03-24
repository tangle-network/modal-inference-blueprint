use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A single Modal model endpoint that this operator serves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEndpoint {
    /// Model identifier (e.g., "cosyvoice3", "fish-s2-pro", "pyannote")
    pub name: String,

    /// Task type: tts, stt, diarize, clone, enhance, translate, langid, speakerid, vad
    #[serde(rename = "type")]
    pub task_type: String,

    /// The Modal deployment URL (e.g., "https://your-org--cosyvoice3-service.modal.run")
    pub modal_endpoint: String,

    /// Health check path (default: /health)
    #[serde(default = "default_health_path")]
    pub health_path: String,

    /// Synthesis path (default: /synthesize for TTS, /transcribe for STT)
    #[serde(default)]
    pub inference_path: Option<String>,

    /// Pricing: cost per 1K units (characters for TTS, seconds for STT)
    #[serde(default)]
    pub price_per_1k: Option<Decimal>,
}

fn default_health_path() -> String {
    "/health".to_string()
}

impl ModelEndpoint {
    /// Returns the inference path based on task type.
    pub fn resolve_inference_path(&self) -> &str {
        if let Some(ref p) = self.inference_path {
            return p;
        }
        match self.task_type.as_str() {
            "tts" => "/synthesize",
            "stt" => "/transcribe",
            "diarize" => "/diarize",
            "clone" => "/clone_voice",
            "enhance" => "/enhance",
            "translate" => "/translate",
            "langid" => "/identify-language",
            "speakerid" => "/identify",
            "vad" => "/detect",
            _ => "/synthesize",
        }
    }
}

/// Operator configuration — loaded from config/operator.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Operator display name
    pub name: String,

    /// Tangle marketplace registration
    #[serde(default)]
    pub gateway: GatewayConfig,

    /// Tangle network settings
    #[serde(default)]
    pub tangle: TangleConfig,

    /// QoS / heartbeat settings
    #[serde(default)]
    pub qos: QoSSettings,

    /// HTTP server settings
    #[serde(default)]
    pub server: ServerConfig,

    /// Cost management
    #[serde(default)]
    pub cost: CostConfig,

    /// Model endpoints this operator serves
    #[serde(default)]
    pub models: Vec<ModelEndpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostConfig {
    /// Stop Modal apps after this many minutes of no requests. 0 = disabled.
    #[serde(default = "default_idle_shutdown")]
    pub idle_shutdown_minutes: u64,

    /// How often to check for idle models (minutes).
    #[serde(default = "default_idle_check")]
    pub idle_check_interval_minutes: u64,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            idle_shutdown_minutes: 30,
            idle_check_interval_minutes: 5,
        }
    }
}

fn default_idle_shutdown() -> u64 { 30 }
fn default_idle_check() -> u64 { 5 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GatewayConfig {
    /// Tangle marketplace API URL
    #[serde(default = "default_gateway_url")]
    pub url: String,

    /// API key for marketplace registration (obtained from marketplace)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Payout email for revenue share
    #[serde(default)]
    pub payout_email: Option<String>,

    /// Stripe Connect account ID (for fiat payouts)
    #[serde(default)]
    pub stripe_connect_id: Option<String>,
}

fn default_gateway_url() -> String {
    "https://api.marketplace".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TangleConfig {
    pub service_id: Option<u64>,
    pub blueprint_id: u64,
    #[serde(default)]
    pub rpc_url: String,
    #[serde(default)]
    pub operator_key: String,
    #[serde(default)]
    pub status_registry_address: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSSettings {
    /// Heartbeat interval in seconds
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,

    /// Metrics collection interval in seconds
    #[serde(default = "default_metrics_interval")]
    pub metrics_interval_secs: u64,

    /// Prometheus metrics port
    #[serde(default = "default_prometheus_port")]
    pub prometheus_port: u16,
}

impl Default for QoSSettings {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: 30,
            metrics_interval_secs: 60,
            prometheus_port: 9090,
        }
    }
}

fn default_heartbeat_interval() -> u64 { 30 }
fn default_metrics_interval() -> u64 { 60 }
fn default_prometheus_port() -> u16 { 9090 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_host")]
    pub host: String,

    /// Max concurrent requests per model
    #[serde(default = "default_concurrency")]
    pub max_concurrency: usize,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "0.0.0.0".to_string(),
            max_concurrency: 10,
            timeout_secs: 120,
        }
    }
}

fn default_port() -> u16 { 8080 }
fn default_host() -> String { "0.0.0.0".to_string() }
fn default_concurrency() -> usize { 10 }
fn default_timeout() -> u64 { 120 }

impl OperatorConfig {
    /// Load config from file or default path.
    pub fn load(path: Option<PathBuf>) -> anyhow::Result<Self> {
        let path = path.unwrap_or_else(|| PathBuf::from("config/operator.toml"));

        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let config: OperatorConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            anyhow::bail!("Config file not found: {}", path.display())
        }
    }
}
