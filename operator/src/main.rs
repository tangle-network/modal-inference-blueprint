use alloy_sol_types::SolValue;
use modal_inference::{config::OperatorConfig, ModalInferenceServer};
fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    assert!(
        config.gpu.expected_gpu_count > 0 && config.gpu.min_vram_mib > 0,
        "set remote GPU facts from the SDK lease receipt"
    );
    let models: Vec<String> = config
        .modal
        .models
        .iter()
        .map(|model| model.name.clone())
        .collect();
    let tasks: Vec<String> = config
        .modal
        .models
        .iter()
        .map(|model| model.task_type.clone())
        .collect();
    (
        models,
        tasks,
        config.gpu.expected_gpu_count,
        config.gpu.min_vram_mib,
        config
            .gpu
            .gpu_model
            .clone()
            .expect("GPU catalog class required"),
        format!("http://{}:{}", config.server.host, config.server.port),
    )
        .abi_encode()
}
tangle_inference_core::run_operator!(
    OperatorConfig::load(None),
    registration_payload,
    modal_inference::router(),
    |config| ModalInferenceServer { config },
    qos
);
