//! Modal-specific operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`)
//! lives in `tangle-inference-core` and is re-exported here. The `modal` section
//! carries all task-type pricing and the list of Modal endpoints this operator
//! serves.

use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use tangle_inference_core::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Human-readable operator name (shown in /health).
    #[serde(default = "default_name")]
    pub name: String,

    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// HTTP server configuration (shared).
    #[serde(default = "default_server_config")]
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration (shared).
    pub billing: BillingConfig,

    /// Modal-specific backend configuration (task-type pricing, model list,
    /// idle shutdown settings).
    pub modal: ModalConfig,
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional — disabled by default).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

fn default_name() -> String {
    "modal-operator".to_string()
}

fn default_server_config() -> ServerConfig {
    serde_json::from_str("{}").expect("ServerConfig defaults are valid")
}

/// Modal backend configuration — the only section that's truly modal-specific.
///
/// Holds per-task-type pricing (dispatched via `TaskTypeCostModel`), the list
/// of Modal endpoints this operator serves, and idle-shutdown settings for
/// cost optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalConfig {
    // Task-type pricing (base token units).
    /// Per input token (chat, text generation).
    #[serde(default)]
    pub price_per_input_token: u64,
    /// Per output token (chat, text generation).
    #[serde(default)]
    pub price_per_output_token: u64,
    /// Per 1,000 characters (TTS, voice cloning, voice design).
    #[serde(default)]
    pub price_per_1k_tts_chars: u64,
    /// Per second of audio (STT, diarize, translate, langid, vad, s2s, speakerid).
    #[serde(default)]
    pub price_per_stt_second: u64,
    /// Per image (image generation).
    #[serde(default)]
    pub price_per_image: u64,
    /// Per second of generated video.
    #[serde(default)]
    pub price_per_video_second: u64,
    /// Per second of generated music.
    #[serde(default)]
    pub price_per_music_second: u64,
    /// Per 1K embedding tokens.
    #[serde(default)]
    pub price_per_1k_embedding_tokens: u64,
    /// Flat price per request for fixed-cost jobs (stitch, enhance, convert).
    #[serde(default)]
    pub default_price_per_request: u64,

    /// List of Modal endpoints this operator serves.
    #[serde(default)]
    pub models: Vec<ModelEndpoint>,
}

impl Default for ModalConfig {
    fn default() -> Self {
        Self {
            price_per_input_token: 0,
            price_per_output_token: 0,
            price_per_1k_tts_chars: 0,
            price_per_stt_second: 0,
            price_per_image: 0,
            price_per_video_second: 0,
            price_per_music_second: 0,
            price_per_1k_embedding_tokens: 0,
            default_price_per_request: 0,
            models: Vec::new(),
        }
    }
}

/// A single Modal model endpoint that this operator serves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEndpoint {
    /// Model identifier (e.g., "cosyvoice3", "fish-s2-pro", "pyannote").
    pub name: String,

    /// Task type. Determines which `CostModel` sub-entry of `TaskTypeCostModel`
    /// the request is routed to. Canonical values: `chat`, `tts`, `stt`,
    /// `image`, `video`, `music`, `embedding`. Any other value falls through
    /// to the default (flat per-request) cost model.
    #[serde(rename = "type")]
    pub task_type: String,

    /// Modal deployment URL (e.g., "https://your-org--cosyvoice3-service.modal.run").
    pub modal_endpoint: String,

    /// Health check path (default: /health).
    #[serde(default = "default_health_path")]
    pub health_path: String,

    /// Inference path override. Defaults are task-type-specific; see
    /// [`ModelEndpoint::resolve_inference_path`].
    #[serde(default)]
    pub inference_path: Option<String>,
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
            "chat" | "text-generation" | "text" => "/v1/chat/completions",
            "image" | "image-generation" => "/v1/images/generations",
            "video" | "video-generation" => "/v1/videos/generations",
            "video-avatar" => "/generate",
            "video-lipsync" => "/lipsync",
            "video-stitch" => "/stitch",
            "video-understanding" => "/v1/video/analyze",
            "s2s" => "/v1/audio/speech",
            "music" | "music-generation" => "/v1/audio/generations",
            "embedding" => "/v1/embeddings",
            "rerank" => "/v1/rerank",
            "voice-conversion" => "/v1/audio/convert",
            "audio-processing" => "/v1/audio/enhance",
            _ => "/v1/inference",
        }
    }
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        } else if std::path::Path::new("config/operator.toml").exists() {
            builder = builder.add_source(config::File::with_name("config/operator.toml"));
        }

        // Env vars override file config. Prefix: MODAL_OP_ (e.g. MODAL_OP_TANGLE__RPC_URL).
        builder = builder.add_source(
            config::Environment::with_prefix("MODAL_OP")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}
