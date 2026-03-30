use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fmt;

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
            "text-generation" | "text" => "/v1/chat/completions",
            "image-generation" | "image" => "/v1/images/generations",
            "video-generation" | "video" => "/v1/videos/generations",
            "video-avatar" => "/generate",
            "video-lipsync" => "/lipsync",
            "video-stitch" => "/stitch",
            "video-understanding" => "/v1/video/analyze",
            "s2s" => "/v1/audio/speech",
            "music-generation" | "music" => "/v1/audio/generations",
            "embedding" => "/v1/embeddings",
            "rerank" => "/v1/rerank",
            "voice-conversion" => "/v1/audio/convert",
            "audio-processing" => "/v1/audio/enhance",
            _ => "/v1/inference",
        }
    }
}

/// Operator configuration — loaded from config/operator.toml
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Operator display name
    pub name: String,

    /// Tangle marketplace registration
    #[serde(default)]
    pub gateway: GatewayConfig,

    /// Tangle network settings (for QoS / heartbeat / registry)
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

    /// Billing / ShieldedCredits configuration (optional — disabled by default)
    #[serde(default)]
    pub billing: BillingConfig,

    /// Model endpoints this operator serves
    #[serde(default)]
    pub models: Vec<ModelEndpoint>,
}

impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("name", &self.name)
            .field("gateway", &self.gateway)
            .field("tangle", &self.tangle)
            .field("qos", &self.qos)
            .field("server", &self.server)
            .field("cost", &self.cost)
            .field("billing", &self.billing)
            .field("models", &self.models.len())
            .finish()
    }
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

// ---------------------------------------------------------------------------
// Billing / ShieldedCredits
// ---------------------------------------------------------------------------

/// Tangle chain config used exclusively by billing (separate from QoS TangleConfig).
#[derive(Clone, Serialize, Deserialize)]
pub struct BillingTangleConfig {
    /// JSON-RPC endpoint for the Tangle EVM chain
    #[serde(default)]
    pub rpc_url: String,

    /// Chain ID
    #[serde(default)]
    pub chain_id: u64,

    /// Operator private key (hex, with or without 0x prefix).
    /// In production, use a KMS or hardware signer instead.
    #[serde(default)]
    pub operator_key: String,

    /// ShieldedCredits contract address
    #[serde(default)]
    pub shielded_credits: String,

    /// Blueprint ID this operator is registered for
    #[serde(default)]
    pub blueprint_id: u64,

    /// Service ID (set after service activation)
    #[serde(default)]
    pub service_id: Option<u64>,
}

impl fmt::Debug for BillingTangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BillingTangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("shielded_credits", &self.shielded_credits)
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

impl Default for BillingTangleConfig {
    fn default() -> Self {
        Self {
            rpc_url: String::new(),
            chain_id: 0,
            operator_key: String::new(),
            shielded_credits: String::new(),
            blueprint_id: 0,
            service_id: None,
        }
    }
}

/// Per-task-type pricing in tsUSD base units (6 decimals: 1 = 0.000001 tsUSD).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingConfig {
    /// Price per 1,000 characters (TTS, clone, enhance)
    #[serde(default)]
    pub price_per_1k_chars: u64,

    /// Price per second of audio (STT, diarize, translate, langid, vad)
    #[serde(default)]
    pub price_per_second: u64,

    /// Price per image generated
    #[serde(default)]
    pub price_per_image: u64,

    /// Price per input token (text generation fallback)
    #[serde(default)]
    pub price_per_input_token: u64,

    /// Price per output token (text generation fallback)
    #[serde(default)]
    pub price_per_output_token: u64,
}

impl Default for PricingConfig {
    fn default() -> Self {
        Self {
            price_per_1k_chars: 0,
            price_per_second: 0,
            price_per_image: 0,
            price_per_input_token: 0,
            price_per_output_token: 0,
        }
    }
}

/// Billing / ShieldedCredits configuration.
///
/// When `required` is false (default), operators can run without billing —
/// no SpendAuth validation is performed and all requests are served free.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Whether billing (SpendAuth) is required on every request.
    /// When true, requests without a valid SpendAuth are rejected with 402.
    #[serde(default)]
    pub required: bool,

    /// Tangle chain settings for on-chain billing calls.
    #[serde(default)]
    pub tangle: BillingTangleConfig,

    /// Per-task-type pricing.
    #[serde(default)]
    pub pricing: PricingConfig,

    /// Maximum amount a single SpendAuth can authorize (anti-abuse).
    #[serde(default)]
    pub max_spend_per_request: u64,

    /// Minimum balance required in a credit account to serve a request.
    #[serde(default)]
    pub min_credit_balance: u64,

    /// Minimum charge amount per request (gas cost protection).
    /// Requests whose pre-authorized amount is below this are rejected.
    #[serde(default)]
    pub min_charge_amount: u64,

    /// Maximum retries for claim_payment on-chain calls.
    #[serde(default = "default_claim_max_retries")]
    pub claim_max_retries: u32,

    /// Clock skew tolerance in seconds for SpendAuth expiry checks.
    #[serde(default = "default_clock_skew_tolerance")]
    pub clock_skew_tolerance_secs: u64,

    /// Maximum gas price in gwei the operator is willing to pay for billing txs.
    /// 0 = no cap (default).
    #[serde(default)]
    pub max_gas_price_gwei: u64,

    /// Path to persist used nonces across restarts (replay protection).
    /// Defaults to `data/nonces.json`. Without persistence, nonces are lost on
    /// restart, allowing replay of unexpired SpendAuth signatures.
    #[serde(default = "default_nonce_store_path")]
    pub nonce_store_path: Option<std::path::PathBuf>,

    /// ERC-20 token address for x402 payment (e.g. tsUSD).
    /// Included in 402 Payment Required responses so clients know which token to use.
    #[serde(default)]
    pub payment_token_address: Option<String>,
}

impl Default for BillingConfig {
    fn default() -> Self {
        Self {
            required: false,
            tangle: BillingTangleConfig::default(),
            pricing: PricingConfig::default(),
            max_spend_per_request: 0,
            min_credit_balance: 0,
            min_charge_amount: 0,
            claim_max_retries: default_claim_max_retries(),
            clock_skew_tolerance_secs: default_clock_skew_tolerance(),
            max_gas_price_gwei: 0,
            nonce_store_path: default_nonce_store_path(),
            payment_token_address: None,
        }
    }
}

fn default_claim_max_retries() -> u32 { 3 }
fn default_clock_skew_tolerance() -> u64 { 30 }
fn default_nonce_store_path() -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from("data/nonces.json"))
}

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
