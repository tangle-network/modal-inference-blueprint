use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use tangle_inference_core::qos::QoSConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    #[serde(default = "default_name")]
    pub name: String,

    pub tangle: TangleConfig,

    #[serde(default = "default_server_config")]
    pub server: ServerConfig,

    pub billing: BillingConfig,

    pub modal: ModalConfig,
    #[serde(default)]
    pub gpu: Option<GpuConfig>,

    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

fn default_name() -> String {
    "modal-operator".to_string()
}

fn default_server_config() -> ServerConfig {
    serde_json::from_str("{}").expect("ServerConfig defaults are valid")
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalConfig {
    // Task-type pricing (base token units).
    #[serde(default)]
    pub price_per_input_token: u64,
    #[serde(default)]
    pub price_per_output_token: u64,
    #[serde(default)]
    pub price_per_1k_tts_chars: u64,
    #[serde(default)]
    pub price_per_stt_second: u64,
    #[serde(default)]
    pub price_per_image: u64,
    #[serde(default)]
    pub price_per_video_second: u64,
    #[serde(default)]
    pub price_per_music_second: u64,
    #[serde(default)]
    pub price_per_1k_embedding_tokens: u64,
    #[serde(default)]
    pub default_price_per_request: u64,

    #[serde(default)]
    pub models: Vec<ModelEndpoint>,
    #[serde(default)]
    pub idle_shutdown_minutes: Option<u64>,
    #[serde(default)]
    pub idle_check_interval_minutes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEndpoint {
    pub name: String,

    #[serde(rename = "type")]
    pub task_type: String,

    pub modal_endpoint: String,

    #[serde(default = "default_health_path")]
    pub health_path: String,

    #[serde(default)]
    pub inference_path: Option<String>,
}

fn default_health_path() -> String {
    "/health".to_string()
}

impl ModelEndpoint {
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
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        } else if std::path::Path::new("config/operator.toml").exists() {
            builder = builder.add_source(config::File::with_name("config/operator.toml"));
        }

        for prefix_separator in ["__", "_"] {
            builder = builder.add_source(
                config::Environment::with_prefix("MODAL_OP")
                    .prefix_separator(prefix_separator)
                    .separator("__")
                    .try_parsing(true),
            );
        }

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        let m = &cfg.modal;
        for model in m.models.iter().filter(|_| cfg.billing.billing_required) {
            let price = match crate::server::canonical_task_key(&model.task_type) {
                Some("chat") => m.price_per_input_token.min(m.price_per_output_token),
                Some("tts") => m.price_per_1k_tts_chars,
                Some("stt") => m.price_per_stt_second,
                Some("image") => m.price_per_image,
                Some("video") => m.price_per_video_second,
                Some("music") => m.price_per_music_second,
                Some("embedding") => m.price_per_1k_embedding_tokens,
                _ => m.default_price_per_request,
            };
            anyhow::ensure!(
                price > 0,
                "billing_required needs a nonzero price for model {}",
                model.name
            );
        }
        Ok(cfg)
    }
}
