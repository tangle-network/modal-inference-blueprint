//! Prometheus metrics for the Modal inference operator.
//!
//! These metrics are:
//! 1. Exposed via /metrics for the ph0ny gateway to scrape
//! 2. Encoded as MetricPair[] for on-chain submission (reputation/slashing)

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, Gauge, Opts,
    Registry, TextEncoder,
};

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

pub static REQUEST_COUNT: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let counter = IntCounterVec::new(
        Opts::new("modal_operator_requests_total", "Total inference requests")
            .namespace("ph0ny"),
        &["model", "status"],
    ).expect("metric");
    REGISTRY.register(Box::new(counter.clone())).expect("register");
    counter
});

pub static REQUEST_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    let hist = HistogramVec::new(
        HistogramOpts::new("modal_operator_request_duration_ms", "Request duration in ms")
            .namespace("ph0ny")
            .buckets(vec![10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(hist.clone())).expect("register");
    hist
});

pub static ACTIVE_REQUESTS: LazyLock<Gauge> = LazyLock::new(|| {
    let gauge = Gauge::new("ph0ny_modal_operator_active_requests", "Active requests")
        .expect("metric");
    REGISTRY.register(Box::new(gauge.clone())).expect("register");
    gauge
});

pub static CHARACTERS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let counter = IntCounterVec::new(
        Opts::new("modal_operator_characters_total", "Characters synthesized (TTS)")
            .namespace("ph0ny"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(counter.clone())).expect("register");
    counter
});

pub static AUDIO_SECONDS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let counter = IntCounterVec::new(
        Opts::new("modal_operator_audio_seconds_total", "Audio seconds processed (STT)")
            .namespace("ph0ny"),
        &["model"],
    ).expect("metric");
    REGISTRY.register(Box::new(counter.clone())).expect("register");
    counter
});

// Atomic counters for on-chain metric submission
static TOTAL_SUCCESS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ERROR: AtomicU64 = AtomicU64::new(0);
static LATENCY_SUM_MS: AtomicU64 = AtomicU64::new(0);
static LATENCY_COUNT: AtomicU64 = AtomicU64::new(0);

/// RAII guard for request tracking.
pub struct RequestGuard {
    model: String,
}

impl RequestGuard {
    pub fn new(model: &str) -> Self {
        ACTIVE_REQUESTS.inc();
        Self { model: model.to_string() }
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
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

/// Gather Prometheus metrics as text.
pub fn gather() -> String {
    let encoder = TextEncoder::new();
    let families = REGISTRY.gather();
    let mut buf = Vec::new();
    encoder.encode(&families, &mut buf).unwrap_or_default();
    String::from_utf8(buf).unwrap_or_default()
}

/// Get metrics for on-chain submission.
/// Returns (name, value) pairs compatible with the QoS MetricPair ABI encoding.
pub fn on_chain_metrics() -> Vec<(String, u64)> {
    let success = TOTAL_SUCCESS.load(Ordering::Relaxed);
    let error = TOTAL_ERROR.load(Ordering::Relaxed);
    let total = success + error;
    let latency_count = LATENCY_COUNT.load(Ordering::Relaxed);
    let latency_avg = if latency_count > 0 {
        LATENCY_SUM_MS.load(Ordering::Relaxed) / latency_count
    } else {
        0
    };
    let uptime_bps = if total > 0 {
        (success * 10000) / total
    } else {
        10000 // 100% if no requests yet
    };

    vec![
        ("requests_total".into(), total),
        ("requests_success".into(), success),
        ("requests_error".into(), error),
        ("latency_avg_ms".into(), latency_avg),
        ("uptime_bps".into(), uptime_bps), // basis points: 9950 = 99.50%
    ]
}
