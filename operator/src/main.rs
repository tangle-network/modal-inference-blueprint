use alloy_sol_types::SolValue;
use modal_inference::{config::OperatorConfig, ModalInferenceServer};
fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu = config
        .gpu
        .as_ref()
        .expect("supply remote GPU facts for registration");
    assert!(
        gpu.min_vram_mib > 0,
        "supply VRAM from the provider allocation receipt"
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
        gpu.min_vram_mib,
        gpu.gpu_model.clone().expect("GPU catalog class required"),
        format!("http://{}:{}", config.server.host, config.server.port),
    )
        .abi_encode()
}
tangle_inference_core::run_operator!(
    OperatorConfig::load(None),
    registration_payload,
    |config| modal_inference::router_with_config(config),
    |config| ModalInferenceServer { config },
    qos
);
