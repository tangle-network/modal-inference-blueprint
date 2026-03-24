//! Comprehensive operator metrics for the Modal inference blueprint.
//!
//! Three output surfaces:
//! 1. Prometheus /metrics — scraped by platform gateway
//! 2. On-chain MetricPair[] — submitted via QoS for reputation/slashing
//! 3. /health JSON — consumed by platform health checker for dashboard display

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntCounter,
    Gauge, GaugeVec, Opts, Registry, TextEncoder,
};

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);
static STARTED_AT: LazyLock<Instant> = LazyLock::new(Instant::now);

// ═══════════════════════════════════════════════════════════════════════
// Request metrics
// ═══════════════════════════════════════════════════════════════════════

pub static REQUEST_COUNT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let c = IntCounterVec::new(
        Opts::new("tangle_operator_requests_total", "Total inference requests"),
        &["model", "status"], // status: success | error | timeout
    ).expect("metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

pub static REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let h = HistogramVec::new(
        HistogramOpts::new("tangle_operator_request_duration_ms", "Request duration in ms")
            .buckets(vec![10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0]),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(h.clone())).expect("register");
    h
});

pub static ACTIVE_REQUESTS: LazyLock<Gauge> = LazyLock::new(|| {
    let g = Gauge::new("tangle_operator_active_requests", "Currently processing requests").expect("metric");
    REGISTRY.register(Box::new(g.clone())).expect("register");
    g
});

pub static MAX_CONCURRENT_REQUESTS: LazyLock<Gauge> = LazyLock::new(|| {
    let g = Gauge::new("tangle_operator_max_concurrent", "Peak concurrent requests observed").expect("metric");
    REGISTRY.register(Box::new(g.clone())).expect("register");
    g
});

// ═══════════════════════════════════════════════════════════════════════
// Throughput metrics
// ═══════════════════════════════════════════════════════════════════════

pub static CHARACTERS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let c = IntCounterVec::new(
        Opts::new("tangle_operator_characters_total", "Characters synthesized (TTS)"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

pub static AUDIO_SECONDS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let c = IntCounterVec::new(
        Opts::new("tangle_operator_audio_seconds_total", "Audio seconds processed (STT/diarize)"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

pub static TOKENS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let c = IntCounterVec::new(
        Opts::new("tangle_operator_tokens_total", "LLM tokens generated (S2S/chat)"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

// ═══════════════════════════════════════════════════════════════════════
// Cold start / infrastructure metrics
// ═══════════════════════════════════════════════════════════════════════

pub static COLD_STARTS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let c = IntCounterVec::new(
        Opts::new("tangle_operator_cold_starts_total", "Container cold starts"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

pub static COLD_START_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let h = HistogramVec::new(
        HistogramOpts::new("tangle_operator_cold_start_duration_ms", "Cold start duration")
            .buckets(vec![1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 30000.0, 60000.0]),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(h.clone())).expect("register");
    h
});

pub static MODEL_STATUS: LazyLock<GaugeVec> = LazyLock::new(|| {
    let g = GaugeVec::new(
        Opts::new("tangle_operator_model_status", "Model status (1=running, 0=stopped)"),
        &["model", "task_type"],
    ).expect("metric");
    REGISTRY.register(Box::new(g.clone())).expect("register");
    g
});

// ═══════════════════════════════════════════════════════════════════════
// Quality metrics (populated from periodic benchmarks)
// ═══════════════════════════════════════════════════════════════════════

pub static QUALITY_MOS: LazyLock<GaugeVec> = LazyLock::new(|| {
    let g = GaugeVec::new(
        Opts::new("tangle_operator_quality_mos", "TTS naturalness score (1-5, higher is better)"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(g.clone())).expect("register");
    g
});

pub static QUALITY_WER: LazyLock<GaugeVec> = LazyLock::new(|| {
    let g = GaugeVec::new(
        Opts::new("tangle_operator_quality_wer", "STT word error rate (0-1, lower is better)"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(g.clone())).expect("register");
    g
});

pub static QUALITY_SPEAKER_SIM: LazyLock<GaugeVec> = LazyLock::new(|| {
    let g = GaugeVec::new(
        Opts::new("tangle_operator_quality_speaker_sim", "Voice similarity to reference (0-1)"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(g.clone())).expect("register");
    g
});

// ═══════════════════════════════════════════════════════════════════════
// Heartbeat / uptime
// ═══════════════════════════════════════════════════════════════════════

pub static HEARTBEATS_SENT: LazyLock<IntCounter> = LazyLock::new(|| {
    let c = IntCounter::new("tangle_operator_heartbeats_total", "Heartbeats sent to chain").expect("metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

pub static HEARTBEATS_FAILED: LazyLock<IntCounter> = LazyLock::new(|| {
    let c = IntCounter::new("tangle_operator_heartbeats_failed_total", "Failed heartbeat submissions").expect("metric");
    REGISTRY.register(Box::new(c.clone())).expect("register");
    c
});

// ═══════════════════════════════════════════════════════════════════════
// Atomic counters for on-chain submission
// ═══════════════════════════════════════════════════════════════════════

static TOTAL_SUCCESS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERROR: AtomicU64 = AtomicU64::new(0);
static TOTAL_TIMEOUT: AtomicU64 = AtomicU64::new(0);
static LATENCY_SUM_MS: AtomicU64 = AtomicU64::new(0);
static LATENCY_COUNT: AtomicU64 = AtomicU64::new(0);
static LATENCY_MAX_MS: AtomicU64 = AtomicU64::new(0);
static CHARS_SYNTHESIZED: AtomicU64 = AtomicU64::new(0);
static AUDIO_SECONDS: AtomicU64 = AtomicU64::new(0);
static COLD_START_COUNT: AtomicU64 = AtomicU64::new(0);
static PEAK_CONCURRENT: AtomicU64 = AtomicU64::new(0);

// ═══════════════════════════════════════════════════════════════════════
// Request guard — RAII tracking
// ═══════════════════════════════════════════════════════════════════════

pub struct RequestGuard {
    model: String,
    start: Instant,
    was_cold: bool,
}

impl RequestGuard {
    pub fn new(model: &str) -> Self {
        // Init startup time
        let _ = *STARTED_AT;

        ACTIVE_REQUESTS.inc();
        let current = ACTIVE_REQUESTS.get() as u64;
        let peak = PEAK_CONCURRENT.load(Ordering::Relaxed);
        if current > peak {
            PEAK_CONCURRENT.store(current, Ordering::Relaxed);
            MAX_CONCURRENT_REQUESTS.set(current as f64);
        }

        Self {
            model: model.to_string(),
            start: Instant::now(),
            was_cold: false,
        }
    }

    /// Mark this request as hitting a cold container
    pub fn mark_cold_start(&mut self, cold_start_ms: u64) {
        self.was_cold = true;
        COLD_STARTS.with_label_values(&[&self.model]).inc();
        COLD_START_DURATION.with_label_values(&[&self.model]).observe(cold_start_ms as f64);
        COLD_START_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    /// Record characters synthesized (TTS)
    pub fn record_chars(&self, chars: u64) {
        CHARACTERS_TOTAL.with_label_values(&[&self.model]).inc_by(chars);
        CHARS_SYNTHESIZED.fetch_add(chars, Ordering::Relaxed);
    }

    /// Record audio seconds processed (STT/diarize)
    pub fn record_audio_seconds(&self, seconds: u64) {
        AUDIO_SECONDS_TOTAL.with_label_values(&[&self.model]).inc_by(seconds);
        AUDIO_SECONDS.fetch_add(seconds, Ordering::Relaxed);
    }

    pub fn finish(self, success: bool, latency_ms: u64) {
        let status = if success { "success" } else { "error" };
        REQUEST_COUNT.with_label_values(&[&self.model, status]).inc();
        REQUEST_DURATION.with_label_values(&[&self.model]).observe(latency_ms as f64);

        if success {
            TOTAL_SUCCESS.fetch_add(1, Ordering::Relaxed);
        } else {
            TOTAL_ERROR.fetch_add(1, Ordering::Relaxed);
        }
        LATENCY_SUM_MS.fetch_add(latency_ms, Ordering::Relaxed);
        LATENCY_COUNT.fetch_add(1, Ordering::Relaxed);

        // Track max latency
        let prev_max = LATENCY_MAX_MS.load(Ordering::Relaxed);
        if latency_ms > prev_max {
            LATENCY_MAX_MS.store(latency_ms, Ordering::Relaxed);
        }
    }

    pub fn finish_timeout(self) {
        REQUEST_COUNT.with_label_values(&[&self.model, "timeout"]).inc();
        TOTAL_TIMEOUT.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Output: Prometheus text
// ═══════════════════════════════════════════════════════════════════════

pub fn gather() -> String {
    let encoder = TextEncoder::new();
    let families = REGISTRY.gather();
    let mut buf = Vec::new();
    encoder.encode(&families, &mut buf).unwrap_or_default();
    String::from_utf8(buf).unwrap_or_default()
}

// ═══════════════════════════════════════════════════════════════════════
// Output: on-chain metrics (for QoS submission)
// ═══════════════════════════════════════════════════════════════════════

pub fn on_chain_metrics() -> Vec<(String, u64)> {
    let success = TOTAL_SUCCESS.load(Ordering::Relaxed);
    let error = TOTAL_ERROR.load(Ordering::Relaxed);
    let timeout = TOTAL_TIMEOUT.load(Ordering::Relaxed);
    let total = success + error + timeout;
    let latency_count = LATENCY_COUNT.load(Ordering::Relaxed);
    let latency_avg = if latency_count > 0 {
        LATENCY_SUM_MS.load(Ordering::Relaxed) / latency_count
    } else { 0 };
    let latency_max = LATENCY_MAX_MS.load(Ordering::Relaxed);
    let uptime_bps = if total > 0 {
        (success * 10000) / total
    } else { 10000 };
    let error_rate_bps = if total > 0 {
        ((error + timeout) * 10000) / total
    } else { 0 };

    let uptime_secs = STARTED_AT.elapsed().as_secs();

    vec![
        ("requests_total".into(), total),
        ("requests_success".into(), success),
        ("requests_error".into(), error),
        ("requests_timeout".into(), timeout),
        ("latency_avg_ms".into(), latency_avg),
        ("latency_max_ms".into(), latency_max),
        ("uptime_bps".into(), uptime_bps),
        ("error_rate_bps".into(), error_rate_bps),
        ("chars_synthesized".into(), CHARS_SYNTHESIZED.load(Ordering::Relaxed)),
        ("audio_seconds".into(), AUDIO_SECONDS.load(Ordering::Relaxed)),
        ("cold_starts".into(), COLD_START_COUNT.load(Ordering::Relaxed)),
        ("peak_concurrent".into(), PEAK_CONCURRENT.load(Ordering::Relaxed)),
        ("uptime_seconds".into(), uptime_secs),
    ]
}

// ═══════════════════════════════════════════════════════════════════════
// Output: health JSON (for platform health checker)
// ═══════════════════════════════════════════════════════════════════════

pub fn health_summary() -> serde_json::Value {
    let success = TOTAL_SUCCESS.load(Ordering::Relaxed);
    let error = TOTAL_ERROR.load(Ordering::Relaxed);
    let timeout = TOTAL_TIMEOUT.load(Ordering::Relaxed);
    let total = success + error + timeout;
    let latency_count = LATENCY_COUNT.load(Ordering::Relaxed);

    serde_json::json!({
        "uptime_seconds": STARTED_AT.elapsed().as_secs(),
        "requests": {
            "total": total,
            "success": success,
            "error": error,
            "timeout": timeout,
        },
        "latency": {
            "avg_ms": if latency_count > 0 { LATENCY_SUM_MS.load(Ordering::Relaxed) / latency_count } else { 0 },
            "max_ms": LATENCY_MAX_MS.load(Ordering::Relaxed),
        },
        "throughput": {
            "chars_synthesized": CHARS_SYNTHESIZED.load(Ordering::Relaxed),
            "audio_seconds": AUDIO_SECONDS.load(Ordering::Relaxed),
        },
        "infrastructure": {
            "cold_starts": COLD_START_COUNT.load(Ordering::Relaxed),
            "peak_concurrent": PEAK_CONCURRENT.load(Ordering::Relaxed),
            "active_requests": ACTIVE_REQUESTS.get() as u64,
        },
    })
}
