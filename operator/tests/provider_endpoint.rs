//! The packaged config boots the real CLI and proxies a request to a provider endpoint.
use std::{process::Stdio, time::Duration};
use wiremock::{matchers::{method, path}, Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn example_config_boots_cli_and_serves_provider_endpoint() {
    let provider = MockServer::start().await;
    Mock::given(method("GET")).and(path("/health"))
        .respond_with(ResponseTemplate::new(200)).mount(&provider).await;
    Mock::given(method("POST")).and(path("/synthesize"))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type","audio/wav").set_body_bytes(b"RIFFprovider-audio".to_vec()))
        .expect(1).mount(&provider).await;
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let temporary = tempfile::tempdir().unwrap();
    std::fs::create_dir(temporary.path().join("config")).unwrap();
    let example = include_str!("../../config/example.toml")
        .replace("port = 8080", &format!("port = {port}"))
        .replace("billing_required = true", "billing_required = false")
        .replace("https://allocated-endpoint.example.com", &provider.uri());
    std::fs::write(temporary.path().join("config/operator.toml"),example).unwrap();
    let mut process = tokio::process::Command::new(env!("CARGO_BIN_EXE_standalone"))
        .current_dir(temporary.path())
        .env("MODAL_OP_TANGLE__OPERATOR_KEY", "0000000000000000000000000000000000000000000000000000000000000001")
        .env("RUST_LOG","error").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
        .kill_on_drop(true).spawn().unwrap();
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let mut ready=false;
    for _ in 0..100 {
        assert!(process.try_wait().unwrap().is_none(),"standalone exited before serving the example config");
        if client.get(format!("{base}/health")).send().await.is_ok_and(|response|response.status().is_success()) {ready=true;break;}
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(ready,"standalone did not become healthy");
    let response=client.post(format!("{base}/v1/audio/speech"))
        .json(&serde_json::json!({"model":"kokoro-tts","input":"hello","voice":"af_heart"})).send().await.unwrap();
    assert_eq!(response.status(),200);
    assert_eq!(&response.bytes().await.unwrap()[..],b"RIFFprovider-audio");
    process.kill().await.unwrap();
}
