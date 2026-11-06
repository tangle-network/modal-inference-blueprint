//! A real Tangle producer and consumer carry the existing job ABI to provider HTTP.
use alloy_primitives::Bytes;
use alloy_sol_types::SolValue;
use anyhow::{ensure, Result};
use blueprint_anvil_testing_utils::BlueprintHarness;
use modal_inference::{InferenceRequest, InferenceResult, INFERENCE_JOB};
use std::time::Duration;
use wiremock::{
    matchers::{body_partial_json, method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn existing_chain_job_returns_provider_output() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(120), async {
        let provider = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/synthesize"))
            .and(body_partial_json(
                serde_json::json!({"text":"hello","voice_id":"default","format":"wav"}),
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/wav")
                    .set_body_bytes(b"RIFFchain-provider-audio".to_vec()),
            )
            .expect(1)
            .mount(&provider)
            .await;
        let config: modal_inference::config::OperatorConfig =
            toml::from_str(include_str!("../../config/example.toml"))?;
        let mut config = config;
        config.modal.models[0].name = "voice".into();
        config.modal.models[0].modal_endpoint = provider.uri();
        let harness = BlueprintHarness::builder(modal_inference::router_with_config(
            std::sync::Arc::new(config),
        ))
        .poll_interval(Duration::from_millis(50))
        .spawn()
        .await?;
        let request = InferenceRequest {
            model: "voice".into(),
            inputData: Bytes::from_static(b"hello"),
            inputType: "text".into(),
            outputType: "audio".into(),
        };
        let call = harness
            .submit_job(INFERENCE_JOB, Bytes::from(request.abi_encode()))
            .await?;
        let output = harness.wait_for_job_result_on_chain(call).await?;
        let result = InferenceResult::abi_decode(&output)?;
        harness.shutdown().await;
        ensure!(
            &result.outputData[..] == b"RIFFchain-provider-audio",
            "chain job returned empty or fabricated output"
        );
        ensure!(
            result.unitsConsumed == 5,
            "text input units were not measured"
        );
        provider.verify().await;
        Ok(())
    })
    .await?
}
