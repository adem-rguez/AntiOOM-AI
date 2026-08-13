mod grpc;
mod http;
mod profiler;
mod registry;
mod session;
mod vram;

use std::net::SocketAddr;
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::AtomicU16;
use llama_backend::LlamaBackend;
use sd_backend::SdBackend;
use whisper_backend::WhisperBackend;
use tts_backend::TtsBackend;
use video_backend::VideoBackend;
use moe_cache::MoeExpertCache;
use pool_protocol::{ClusterPoolManager, PeerNode};
use proto::daemon_service_server::DaemonServiceServer;
use registry::BackendRegistry;
use session::SessionManager;
use tracing::info;
use vram::VramArbiter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("Starting AIATM Local Inference Daemon v{}", env!("CARGO_PKG_VERSION"));

    // 1. Probe Hardware & Initialize VRAM Arbiter
    let hardware_info = profiler::HardwareProfiler::probe();
    let vram_arbiter = Arc::new(VramArbiter::new(hardware_info.total_vram_bytes));

    // 2. Initialize Session Manager, MoE Cache, and Cluster Pool Manager
    let session_manager = Arc::new(SessionManager::new());
    let moe_cache = Arc::new(MoeExpertCache::new(4));
    let cluster_pool = Arc::new(ClusterPoolManager::new("local-node-primary".to_string()));

    // Register primary node
    cluster_pool
        .register_peer(PeerNode {
            node_id: "local-node-primary".to_string(),
            address: "127.0.0.1:50051".to_string(),
            free_vram_bytes: hardware_info.free_vram_bytes,
            total_vram_bytes: hardware_info.total_vram_bytes,
            latency_ms: 1,
        })
        .await;

    // 3. Initialize Backend Registry & Register plugins across all modalities
    let registry = Arc::new(BackendRegistry::new());

    registry.register_backend(Box::new(LlamaBackend::new())).await;
    info!("Registered backend plugin: llama.cpp (Text, Embedding)");

    registry.register_backend(Box::new(SdBackend::new())).await;
    info!("Registered backend plugin: stable-diffusion.cpp (Image)");

    registry.register_backend(Box::new(WhisperBackend::new())).await;
    info!("Registered backend plugin: whisper.cpp (Audio ASR)");

    registry.register_backend(Box::new(TtsBackend::new())).await;
    info!("Registered backend plugin: Kokoro TTS (Audio TTS)");

    registry.register_backend(Box::new(VideoBackend::new())).await;
    info!("Registered backend plugin: Wan Video Runner (Video)");

    // 4. Start HTTP OpenAI-compatible server & Dashboard on 0.0.0.0:8080
    let loaded_models = Arc::new(tokio::sync::Mutex::new(HashMap::<String, http::LoadedModelEntry>::new()));
    let next_port = Arc::new(AtomicU16::new(50052));
    let active_model = Arc::new(tokio::sync::Mutex::new(http::LoadedModelStatus {
        is_loaded: false,
        model_path: None,
        gpu_layers: None,
        context_size: None,
    }));

    let http_state = http::AppState {
        registry: registry.clone(),
        session_manager: session_manager.clone(),
        moe_cache: moe_cache.clone(),
        cluster_pool: cluster_pool.clone(),
        loaded_models,
        next_port,
        active_model,
    };
    let app = http::create_router(http_state);
    let http_addr: SocketAddr = "0.0.0.0:8080".parse()?;

    tokio::spawn(async move {
        info!("HTTP REST server & Web Dashboard listening on http://{}", http_addr);
        let listener = tokio::net::TcpListener::bind(http_addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    // 5. Start gRPC Daemon Service on 0.0.0.0:50051
    let grpc_addr: SocketAddr = "0.0.0.0:50051".parse()?;
    let grpc_service = grpc::DaemonGrpcService::new(registry.clone(), vram_arbiter.clone());

    info!("gRPC Daemon server listening on {}", grpc_addr);
    tonic::transport::Server::builder()
        .add_service(DaemonServiceServer::new(grpc_service))
        .serve(grpc_addr)
        .await?;

    Ok(())
}
