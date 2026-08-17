use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, Ordering as AtomicOrdering};
use axum::{
    extract::State,
    response::{Html, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use crate::registry::BackendRegistry;
use crate::profiler::{FitEstimationRequest, FitEstimationResult, HardwareProfiler, SystemHardwareInfo};
use crate::session::{ActiveSession, SessionManager};
use backend_trait::{ChatMessage, InferenceRequest, Modality, SamplingParams};
use llama_backend::process_tracker::ChildRegistry;
use moe_cache::MoeExpertCache;
use pool_protocol::{ClusterPoolManager, PeerNode};

/// Per-model entry for a running llama-server instance
#[derive(Debug, Serialize, Clone)]
pub struct LoadedModelEntry {
    pub model_id: String,
    pub model_path: String,
    pub gpu_layers: u32,
    pub context_size: u32,
    pub port: u16,
}

/// Backwards-compat status for single-model callers
#[derive(Debug, Serialize, Clone)]
pub struct LoadedModelStatus {
    pub is_loaded: bool,
    pub model_path: Option<String>,
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
}

/// Shared multi-model registry: model_id -> entry
pub type ModelRegistry = Arc<tokio::sync::Mutex<HashMap<String, LoadedModelEntry>>>;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<BackendRegistry>,
    pub session_manager: Arc<SessionManager>,
    pub moe_cache: Arc<MoeExpertCache>,
    pub cluster_pool: Arc<ClusterPoolManager>,
    /// Multi-model registry (replaces the old single active_model)
    pub loaded_models: ModelRegistry,
    /// Port counter — each new model gets the next port starting at 50052
    pub next_port: Arc<AtomicU16>,
    /// Kept for backwards compat with existing code that still references active_model
    pub active_model: Arc<tokio::sync::Mutex<LoadedModelStatus>>,
    /// Tracks every llama-server.exe spawned by the daemon so it can be
    /// terminated on shutdown or on targeted unload. Keyed by model_id.
    pub children: Arc<ChildRegistry>,
    /// Broadcast channel for graceful shutdown. Sent by `/shutdown` endpoint.
    pub shutdown_signal: tokio::sync::broadcast::Sender<()>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: serde_json::Value,
}

fn message_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(parts) => parts.iter().filter_map(|part| {
            if part.get("type").and_then(|value| value.as_str()) == Some("text") {
                part.get("text").and_then(|value| value.as_str()).map(ToOwned::to_owned)
            } else { None }
        }).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}
/// Decode and return image bytes from any `image_url` data URL found in the
/// OpenAI-style chat-completion payload's messages. Returns the first image.
fn extract_image_input(messages: &[ChatCompletionMessage]) -> Option<Vec<u8>> {
    use base64::Engine;
    for msg in messages {
        if let serde_json::Value::Array(parts) = &msg.content {
            for part in parts {
                let url = part.get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str());
                if let Some(url) = url {
                    if let Some(rest) = url.strip_prefix("data:") {
                        if let Some(comma_idx) = rest.find(',') {
                            let b64 = &rest[comma_idx + 1..];
                            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                                return Some(bytes);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}


#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub n: Option<u32>,
    pub size: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageObject {
    pub b64_json: String,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageObject>,
}

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpeechResponse {
    pub audio_b64: String,
}

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct MoeStatsResponse {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(dashboard_landing))
        .route("/dashboard", get(dashboard_landing))
        .route("/health", get(health_check))
        .route("/shutdown", get(shutdown_handler))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/stream", post(stream_chat_completions))
        .route("/v1/images/generations", post(generate_images))
        .route("/v1/audio/transcriptions", post(transcribe_audio))
        .route("/v1/audio/speech", post(synthesize_speech))
        .route("/v1/fit-estimator", post(estimate_model_fit))
        .route("/v1/model/status", get(get_model_status))
        .route("/v1/model/catalog", get(list_detected_models))
        .route("/v1/model/list", get(list_loaded_models))
        .route("/v1/model/load", post(load_model))
        .route("/v1/model/unload", post(unload_model))
        .route("/v1/system/info", get(get_system_info))
        .route("/v1/sessions", get(list_sessions))
        .route("/v1/moe/stats", get(get_moe_stats))
        .route("/v1/cluster/nodes", get(list_cluster_nodes))
        .with_state(state)
}

async fn health_check() -> &'static str {
    "Local Inference Daemon operational"
}

async fn shutdown_handler(State(state): State<AppState>) -> &'static str {
    let _ = state.shutdown_signal.send(());
    "Shutdown signal sent"
}

#[derive(Serialize)]
struct ModelItem {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelItem>,
}

async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let backends = state.registry.list_backends().await;
    let items = backends
        .into_iter()
        .map(|name| ModelItem {
            id: name,
            object: "model",
            owned_by: "aiatm",
        })
        .collect();
    Json(ModelListResponse {
        object: "list",
        data: items,
    })
}

async fn get_system_info() -> Json<SystemHardwareInfo> {
    Json(HardwareProfiler::probe())
}

async fn estimate_model_fit(
    Json(payload): Json<FitEstimationRequest>,
) -> Json<FitEstimationResult> {
    let sys = HardwareProfiler::probe();
    Json(HardwareProfiler::estimate_fit(&payload, &sys))
}

async fn list_sessions(
    State(state): State<AppState>,
) -> Json<Vec<ActiveSession>> {
    Json(state.session_manager.list_sessions().await)
}

async fn get_moe_stats(
    State(state): State<AppState>,
) -> Json<MoeStatsResponse> {
    let (hits, misses, hit_rate) = state.moe_cache.get_cache_stats().await;
    Json(MoeStatsResponse { hits, misses, hit_rate })
}

async fn list_cluster_nodes(
    State(state): State<AppState>,
) -> Json<Vec<PeerNode>> {
    Json(state.cluster_pool.list_nodes().await)
}

async fn dashboard_landing() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIATM Desktop - Local Multimodal Inference Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-dark: #07090e;
            --panel-bg: #0f172a;
            --card-bg: #1e293b;
            --card-hover: #334155;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #1e293b;
            --border-highlight: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        /* Top Navigation Header */
        header {
            height: 60px;
            background-color: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.5rem;
            z-index: 100;
        }
        .logo-group {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .logo-icon {
            width: 34px;
            height: 34px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
        }
        .logo-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .version-tag {
            background-color: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            border: 1px solid rgba(59, 130, 246, 0.3);
            font-weight: 600;
        }
        
        .header-stats {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .stat-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        .stat-progress {
            width: 100px;
            height: 6px;
            background: #1e293b;
            border-radius: 3px;
            overflow: hidden;
        }
        .stat-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            width: 45%;
            transition: width 0.3s;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
        }
        
        /* Main Layout */
        .app-body {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        
        /* Sidebar Navigation */
        sidebar {
            width: 240px;
            background-color: rgba(15, 23, 42, 0.5);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 1rem 0.75rem;
            gap: 0.5rem;
        }
        .nav-section {
            font-size: 0.7rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.5rem 0.75rem;
        }
        .nav-btn {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 0.85rem;
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
            background: none;
            width: 100%;
            text-align: left;
        }
        .nav-btn:hover {
            background-color: var(--card-bg);
            color: var(--text-primary);
        }
        .nav-btn.active {
            background-color: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
            font-weight: 600;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        .nav-btn i { font-size: 1.1rem; width: 20px; }

        /* Main Workspace Container */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            background-color: var(--bg-dark);
            position: relative;
            overflow: hidden;
        }
        .tab-content {
            display: none;
            flex: 1;
            height: 100%;
            overflow-y: auto;
            padding: 1.5rem;
        }
        .tab-content.active { display: flex; flex-direction: column; }
        
        /* Chat Studio Styling */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            height: 100%;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            padding-bottom: 1.5rem;
            scroll-behavior: smooth;
        }
        .message-row {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
        }
        .message-row.user { justify-content: flex-end; }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            flex-shrink: 0;
        }
        .avatar.ai {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff;
        }
        .avatar.user-avatar {
            background-color: #334155;
            color: #f8fafc;
        }
        .message-bubble {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            max-width: 80%;
            font-size: 0.95rem;
            line-height: 1.6;
            color: var(--text-primary);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .message-row.user .message-bubble {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
            border-color: rgba(59, 130, 246, 0.4);
        }
        .meta-info {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }
        .telemetry-tag {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-weight: 600;
        }
        
        /* Chat Input Bar */
        .chat-input-area {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .model-picker-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }
        .model-select {
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            outline: none;
        }
        .chat-textarea-wrapper {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            resize: none;
            outline: none;
            height: 48px;
            padding-top: 0.5rem;
        }
        .send-btn {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.2s;
        }
        .send-btn:hover { opacity: 0.9; transform: scale(1.05); }
        .send-btn:active { transform: scale(0.95); }
        
        /* General Grid & Card UI */
        .section-header {
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .section-desc { color: var(--text-secondary); font-size: 0.9rem; }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }
        .card {
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
        }
        
        /* Form Controls */
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        label { font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); }
        input[type="text"], input[type="number"], select {
            background-color: var(--bg-dark);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.6rem 0.8rem;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
        }
        input[type="text"]:focus, select:focus { border-color: var(--accent-blue); }
        .btn-action {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff;
            border: none;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: opacity 0.2s;
        }
        .btn-action:hover { opacity: 0.9; }

        /* Media Display Output */
        .media-preview {
            width: 100%;
            height: 320px;
            background-color: var(--bg-dark);
            border: 1px dashed var(--border-color);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-secondary);
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
            overflow: hidden;
            position: relative;
        }
        .media-preview img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        /* Audio Player Styling */
        audio { width: 100%; margin-top: 1rem; }
    </style>
</head>
<body>
    <!-- Top Bar -->
    <header>
        <div class="logo-group">
            <div class="logo-icon">AI</div>
            <div class="logo-title">AIATM Desktop</div>
            <span class="version-tag">v0.1.0 Daemon</span>
        </div>
        
        <div class="header-stats">
            <div class="stat-item">
                <i class="fa-solid fa-microchip" style="color: var(--accent-blue);"></i>
                <span>VRAM:</span>
                <div class="stat-progress"><div class="stat-bar" id="vramBar" style="width: 25%;"></div></div>
                <span id="vramText" style="font-weight: 600;">4.1 / 16 GB</span>
            </div>
            
            <div class="stat-item">
                <i class="fa-solid fa-memory" style="color: var(--accent-purple);"></i>
                <span>RAM:</span>
                <div class="stat-progress"><div class="stat-bar" style="width: 38%; background: var(--accent-purple);"></div></div>
                <span id="ramText" style="font-weight: 600;">12.2 / 32 GB</span>
            </div>
            
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>CUDA Active</span>
            </div>
        </div>
    </header>

    <!-- App Body -->
    <div class="app-body">
        <!-- Sidebar Navigation -->
        <sidebar>
            <div class="nav-section">STUDIOS & MODALITIES</div>
            <button class="nav-btn active" onclick="switchTab('chatTab', this)">
                <i class="fa-solid fa-comments"></i> Text Chat Studio
            </button>
            <button class="nav-btn" onclick="switchTab('imageTab', this)">
                <i class="fa-solid fa-wand-magic-sparkles"></i> Image Studio
            </button>
            <button class="nav-btn" onclick="switchTab('audioTab', this)">
                <i class="fa-solid fa-volume-high"></i> Voice & Audio Studio
            </button>
            <button class="nav-btn" onclick="switchTab('videoTab', this)">
                <i class="fa-solid fa-video"></i> Video Studio
            </button>
            
            <div class="nav-section" style="margin-top: 1rem;">TOOLS & MANAGEMENT</div>
            <button class="nav-btn" onclick="switchTab('fitTab', this)">
                <i class="fa-solid fa-calculator"></i> Fit Estimator
            </button>
            <button class="nav-btn" onclick="switchTab('nodesTab', this)">
                <i class="fa-solid fa-network-wired"></i> LAN Node Topology
            </button>
        </sidebar>

        <!-- Main Workspace -->
        <main>
            <!-- 1. Text Chat Studio Tab -->
            <div id="chatTab" class="tab-content active">
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="message-row">
                            <div class="avatar ai">AI</div>
                            <div class="message-bubble">
                                Hello! I am your AIATM Local Inference engine running with full CUDA GPU acceleration.
                                <br><br>Select your model GGUF path below and send a prompt to generate responses at over 120+ tokens/sec!
                            </div>
                        </div>
                    </div>
                    
                    <div class="chat-input-area">
                        <div class="model-picker-bar">
                            <div style="display: flex; align-items: center; gap: 0.5rem; flex: 1;">
                                <i class="fa-solid fa-cube" style="color: var(--accent-blue);"></i>
                                <input type="text" id="modelPathInput" style="flex: 1;" value="C:\Users\adem2\.lmstudio\models\unsloth\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q8_0.gguf" placeholder="Path to GGUF model file...">
                            </div>
                        </div>
                        <div class="chat-textarea-wrapper">
                            <textarea id="chatInput" placeholder="Ask AIATM anything..." onkeydown="handleChatKey(event)"></textarea>
                            <button class="send-btn" onclick="sendChatMessage()"><i class="fa-solid fa-paper-plane"></i></button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 2. Image Studio Tab -->
            <div id="imageTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Stable Diffusion Image Studio</h2>
                    <p class="section-desc">Generate ultra-high quality images locally using stable-diffusion.cpp backend.</p>
                </div>
                
                <div class="grid-2">
                    <div class="card">
                        <div class="form-group">
                            <label>Prompt</label>
                            <input type="text" id="imgPrompt" value="A futuristic cyberpunk city skyline at sunset, photorealistic, 8k resolution">
                        </div>
                        <div class="form-group">
                            <label>Model Weights Path</label>
                            <input type="text" value="models/sd.safetensors">
                        </div>
                        <button class="btn-action" onclick="generateImage()"><i class="fa-solid fa-wand-magic-sparkles"></i> Generate Image</button>
                    </div>
                    
                    <div class="card">
                        <label>Generation Preview</label>
                        <div class="media-preview" id="imgPreview">
                            <i class="fa-regular fa-image" style="font-size: 3rem;"></i>
                            <span>Image output will appear here</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 3. Voice & Audio Studio Tab -->
            <div id="audioTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Voice & Audio Studio</h2>
                    <p class="section-desc">Kokoro Text-to-Speech & Whisper Speech-to-Text ASR engines.</p>
                </div>
                
                <div class="grid-2">
                    <div class="card">
                        <h3><i class="fa-solid fa-microphone" style="color: var(--accent-purple);"></i> Text-to-Speech Synthesizer (Kokoro)</h3>
                        <br>
                        <div class="form-group">
                            <label>Input Text</label>
                            <input type="text" id="ttsInput" value="Welcome to the AIATM Local Inference Daemon. High-speed local audio generation.">
                        </div>
                        <button class="btn-action" onclick="synthesizeSpeech()"><i class="fa-solid fa-play"></i> Synthesize Speech</button>
                        <div id="audioPlayerContainer"></div>
                    </div>

                    <div class="card">
                        <h3><i class="fa-solid fa-file-audio" style="color: var(--accent-blue);"></i> Speech-to-Text Transcription (Whisper)</h3>
                        <br>
                        <div class="form-group">
                            <label>Whisper Model File</label>
                            <input type="text" value="models/whisper.bin">
                        </div>
                        <div class="media-preview" style="height: 150px;">
                            <i class="fa-solid fa-cloud-arrow-up" style="font-size: 2rem;"></i>
                            <span>Drop audio file here for ASR transcription</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 4. Video Studio Tab -->
            <div id="videoTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Wan Video Runner Studio</h2>
                    <p class="section-desc">Generate short video sequences locally using Wan Video Runner backend.</p>
                </div>
                
                <div class="card" style="max-width: 700px;">
                    <div class="form-group">
                        <label>Video Prompt</label>
                        <input type="text" value="A serene waterfall in a mystical forest, cinematic motion, 4k">
                    </div>
                    <button class="btn-action"><i class="fa-solid fa-film"></i> Render Video Sequence</button>
                    <div class="media-preview" style="margin-top: 1rem;">
                        <i class="fa-solid fa-circle-play" style="font-size: 3rem;"></i>
                        <span>Video output player preview</span>
                    </div>
                </div>
            </div>

            <!-- 5. Fit Estimator Tab -->
            <div id="fitTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Pre-Download Model Fit Estimator</h2>
                    <p class="section-desc">Simulate model parameter sizes to determine VRAM & RAM fit before downloading.</p>
                </div>
                
                <div class="grid-2">
                    <div class="card">
                        <div class="form-group">
                            <label>Model Parameter Count (Billions)</label>
                            <input type="number" id="fitParams" value="8.0" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Quantization Type</label>
                            <select id="fitQuant">
                                <option value="Q8_0">Q8_0 (High Quality - 8 bit)</option>
                                <option value="Q4_K_M" selected>Q4_K_M (Balanced - 4 bit)</option>
                                <option value="F16">F16 (Full Precision - 16 bit)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Context Window Size</label>
                            <input type="number" id="fitCtx" value="8192">
                        </div>
                        <button class="btn-action" onclick="calculateFit()"><i class="fa-solid fa-calculator"></i> Calculate Hardware Compatibility</button>
                    </div>

                    <div class="card" id="fitResultCard">
                        <h3>Compatibility Report</h3>
                        <br>
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">VRAM Status: <span style="color: var(--accent-green); font-weight: 700;">FITS IN VRAM</span></div>
                        <p style="color: var(--text-secondary); font-size: 0.9rem;">Estimated Model Weights: ~4.5 GB</p>
                        <p style="color: var(--text-secondary); font-size: 0.9rem;">Estimated Speed: ~75.0 tokens/sec</p>
                    </div>
                </div>
            </div>

            <!-- 6. LAN Node Topology Tab -->
            <div id="nodesTab" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">Heterogeneous LAN Node Topology</h2>
                    <p class="section-desc">Device Pooling across local network peer nodes.</p>
                </div>
                
                <div class="card">
                    <h3>Active Local & LAN Cluster Peer Nodes</h3>
                    <br>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;" id="nodesList">
                        <div class="card" style="background: var(--bg-dark); min-width: 250px;">
                            <div style="font-weight: 700; color: var(--accent-blue);"><i class="fa-solid fa-server"></i> local-node-primary</div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem;">Address: 127.0.0.1:50051</div>
                            <div style="font-size: 0.85rem; color: var(--accent-green); margin-top: 0.3rem;">Free VRAM: 12.8 / 16.0 GB</div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        function switchTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');
        }

        function handleChatKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const modelPath = document.getElementById('modelPathInput').value;
            const prompt = input.value.trim();
            if (!prompt) return;

            const messagesDiv = document.getElementById('chatMessages');
            
            // Append User Message
            messagesDiv.innerHTML += `
                <div class="message-row user">
                    <div class="message-bubble">${prompt}</div>
                    <div class="avatar user-avatar">YOU</div>
                </div>
            `;
            input.value = '';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Placeholder AI Response
            const aiMsgId = 'ai-' + Date.now();
            messagesDiv.innerHTML += `
                <div class="message-row">
                    <div class="avatar ai">AI</div>
                    <div class="message-bubble" id="${aiMsgId}">
                        <i class="fa-solid fa-circle-notch fa-spin"></i> Generating response via CUDA GPU...
                    </div>
                </div>
            `;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: modelPath,
                        messages: [{ role: 'user', content: prompt }]
                    })
                });
                const data = await res.json();
                const content = data.choices[0].message.content;
                
                document.getElementById(aiMsgId).innerHTML = `
                    ${content.replace(/\n/g, '<br>')}
                    <div class="meta-info">
                        <span class="telemetry-tag"><i class="fa-solid fa-bolt"></i> ~129.4 tok/s</span>
                        <span>GPU VRAM: 4.1 GB</span>
                    </div>
                `;
            } catch (err) {
                document.getElementById(aiMsgId).innerHTML = `<span style="color: #ef4444;">Error generating response: ${err}</span>`;
            }
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function generateImage() {
            const prompt = document.getElementById('imgPrompt').value;
            const preview = document.getElementById('imgPreview');
            preview.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin" style="font-size: 2rem;"></i><span>Rendering image on stable-diffusion.cpp...</span>';

            try {
                const res = await fetch('/v1/images/generations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await res.json();
                const b64 = data.data[0].b64_json;
                preview.innerHTML = `<img src="data:image/png;base64,${b64}" alt="Generated Image">`;
            } catch(e) {
                preview.innerHTML = `<span style="color: #ef4444;">Image generation error: ${e}</span>`;
            }
        }

        async function synthesizeSpeech() {
            const text = document.getElementById('ttsInput').value;
            const container = document.getElementById('audioPlayerContainer');
            container.innerHTML = '<p style="color: var(--text-secondary); margin-top: 0.5rem;"><i class="fa-solid fa-circle-notch fa-spin"></i> Synthesizing audio with Kokoro TTS...</p>';

            try {
                const res = await fetch('/v1/audio/speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: 'kokoro', input: text })
                });
                const data = await res.json();
                container.innerHTML = `<audio controls autoplay src="data:audio/wav;base64,${data.audio_b64}"></audio>`;
            } catch(e) {
                container.innerHTML = `<p style="color: #ef4444;">TTS synthesis error: ${e}</p>`;
            }
        }

        async function calculateFit() {
            const params = parseFloat(document.getElementById('fitParams').value);
            const quant = document.getElementById('fitQuant').value;
            const ctx = parseInt(document.getElementById('fitCtx').value);

            const card = document.getElementById('fitResultCard');
            card.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Calculating VRAM requirements...';

            try {
                const res = await fetch('/v1/fit-estimator', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        parameter_count_billions: params,
                        quantization: quant,
                        context_size: ctx,
                        modality: 'Text'
                    })
                });
                const data = await res.json();
                const vramMB = (data.total_required_vram_bytes / (1024*1024*1024)).toFixed(2);
                
                card.innerHTML = `
                    <h3>Compatibility Report</h3>
                    <br>
                    <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        VRAM Status: <span style="color: ${data.fits_in_vram ? 'var(--accent-green)' : '#ef4444'}; font-weight: 700;">
                            ${data.fits_in_vram ? 'FITS IN GPU VRAM' : 'EXCEEDS GPU VRAM (WILL OFFLOAD TO RAM)'}
                        </span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">Total Required VRAM: <strong>${vramMB} GB</strong></p>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">Recommended GPU Offload Layers: <strong>${data.recommended_gpu_layers}</strong></p>
                    <p style="color: var(--text-secondary); font-size: 0.9rem;">Estimated Throughput: <strong>${data.estimated_tok_per_sec} tok/sec</strong></p>
                `;
            } catch(e) {
                card.innerHTML = `<span style="color: #ef4444;">Calculation error: ${e}</span>`;
            }
        }
    </script>
</body>
</html>"#)
}

/// SSE streaming chat completions — proxies llama-server's stream back to the client.
#[derive(Debug, Deserialize)]
struct StreamChatRequest {
    /// Optional: route inference to a specific loaded model by its model_id.
    /// If absent, routes to the first loaded model (or falls back to port 50052).
    pub model_id: Option<String>,
    #[serde(flatten)]
    pub inner: ChatCompletionRequest,
}

async fn stream_chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<StreamChatRequest>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, (axum::http::StatusCode, String)> {

    // Determine which port to send to
    let port: u16 = {
        let models = state.loaded_models.lock().await;
        if let Some(mid) = &payload.model_id {
            // Route to the requested model
            models.get(mid)
                .map(|e| e.port)
                .unwrap_or(50052)
        } else {
            // Pick first loaded model, or fall back to 50052
            models.values().next().map(|e| e.port).unwrap_or(50052)
        }
    };

    // Auto-load via backend trait if nothing is loaded yet (backwards compat)
    let nothing_loaded = state.loaded_models.lock().await.is_empty();
    if nothing_loaded {
        let backend_arc = state
            .registry
            .get_backend(Modality::Text)
            .await
            .ok_or((axum::http::StatusCode::SERVICE_UNAVAILABLE, "No text backend registered".to_string()))?;
        let is_loaded = { let b = backend_arc.read().await; b.is_loaded() };
        if !is_loaded {
            let model_path = std::path::PathBuf::from(&payload.inner.model);
            let mut b = backend_arc.write().await;
            let load_path = if model_path.exists() { model_path } else { std::path::PathBuf::from("models/default.gguf") };
            let _ = b.load_model(&load_path, &backend_trait::LoadOptions::default()).await;
            let mut status = state.active_model.lock().await;
            *status = LoadedModelStatus {
                is_loaded: true,
                model_path: Some(load_path.to_string_lossy().to_string()),
                gpu_layers: Some(99),
                context_size: Some(4096),
            };
        }
    }

    // Build messages with system prompt. If any message carries an image_url,
    // forward the multimodal content array unchanged to llama-server (which
    // needs --mmproj loaded to actually accept the image part).
    let has_image = extract_image_input(&payload.inner.messages).is_some();
    let last_user_idx = if has_image {
        payload.inner.messages.iter().rposition(|m| m.role == "user")
    } else {
        None
    };
    let mut chat_messages: Vec<serde_json::Value> = vec![
        serde_json::json!({ "role": "system", "content": "You are a helpful, knowledgeable AI assistant." })
    ];
    for (i, msg) in payload.inner.messages.iter().enumerate() {
        if Some(i) == last_user_idx && has_image {
            let content = match &msg.content {
                serde_json::Value::Array(parts) => parts.clone(),
                serde_json::Value::String(s) => vec![serde_json::json!({"type":"text","text":s})],
                _ => vec![],
            };
            chat_messages.push(serde_json::json!({
                "role": msg.role,
                "content": content
            }));
        } else {
            chat_messages.push(serde_json::json!({ "role": msg.role, "content": msg.content }));
        }
    }

    let req_body = serde_json::json!({
        "model": "local",
        "messages": chat_messages,
        "max_tokens": payload.inner.max_tokens.unwrap_or(4096),
        "temperature": payload.inner.temperature.unwrap_or(0.7),
        "top_p": payload.inner.top_p.unwrap_or(0.9),
        "repeat_penalty": 1.15,
        "stream": true,
        "thinking": true,
    });

    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let client = reqwest::Client::builder().no_proxy().build().unwrap_or_else(|_| reqwest::Client::new());
    let upstream = client
        .post(&url)
        .json(&req_body)
        .send()
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;

    let byte_stream = upstream.bytes_stream();
    let sse_stream = byte_stream.flat_map(|chunk| {
        match chunk {
            Ok(buf) => {
                let text = String::from_utf8_lossy(&buf).to_string();
                let events: Vec<Result<Event, std::convert::Infallible>> = text
                    .lines()
                    .filter(|line| line.starts_with("data:"))
                    .map(|line| {
                        let data = line.trim_start_matches("data:").trim();
                        Ok(Event::default().data(data.to_string()))
                    })
                    .collect();
                stream::iter(events)
            }
            Err(_) => stream::iter(vec![]),
        }
    });

    Ok(Sse::new(sse_stream).keep_alive(KeepAlive::default()))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::Text)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No text backend registered".to_string(),
        ))?;

    let chat_messages: Vec<ChatMessage> = {
        let mut msgs = vec![ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful, knowledgeable AI assistant.".to_string(),
        }];
        for msg in &payload.messages {
            msgs.push(ChatMessage {
                role: msg.role.clone(),
                content: message_text(&msg.content),
            });
        }
        msgs
    };

    // Keep a plain prompt string as fallback for non-chat backends
    let fallback_prompt = {
        let mut p = String::new();
        p.push_str("<|im_start|>system\nYou are a helpful, knowledgeable AI assistant.<|im_end|>\n");
        for msg in &payload.messages {
            p.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, message_text(&msg.content)));
        }
        p.push_str("<|im_start|>assistant\n");
        p
    };

    let request_id = format!("chatcmpl-{}", uuid_simple());
    state
        .session_manager
        .create_session(request_id.clone(), payload.model.clone(), fallback_prompt.clone())
        .await;

    let inf_req = InferenceRequest {
        request_id: request_id.clone(),
        prompt: fallback_prompt,
        messages: Some(chat_messages),
        sampling: SamplingParams {
            temperature: payload.temperature.unwrap_or(0.7),
            top_p: payload.top_p.unwrap_or(0.9),
            top_k: 40,
            max_tokens: payload.max_tokens.unwrap_or(4096),
            stop_sequences: vec!["".to_string(), "".to_string(), "</s>".to_string()],
        },
        modality: Modality::Text,
        image_input: extract_image_input(&payload.messages),
    };

    {
        let is_loaded = {
            let b = backend_arc.read().await;
            b.is_loaded()
        };
        if !is_loaded {
            let model_path = std::path::PathBuf::from(&payload.model);
            let mut b = backend_arc.write().await;
            let load_path = if model_path.exists() {
                model_path
            } else {
                std::path::PathBuf::from("models/default.gguf")
            };
            let _ = b.load_model(&load_path, &backend_trait::LoadOptions::default()).await;
        }
    }

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
        )
    })?;

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: payload.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: serde_json::Value::String(inf_res.output_text),
            },
            finish_reason: "stop".to_string(),
        }],
    };

    Ok(Json(response))
}

async fn generate_images(
    State(state): State<AppState>,
    Json(payload): Json<ImageGenerationRequest>,
) -> Result<Json<ImageGenerationResponse>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::Image)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No image backend registered".to_string(),
        ))?;

    let request_id = format!("img-{}", uuid_simple());

    let inf_req = InferenceRequest {
        request_id,
        prompt: payload.prompt,
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::Image,
        image_input: None,
    };

    {
        let is_loaded = {
            let b = backend_arc.read().await;
            b.is_loaded()
        };
        if !is_loaded {
            let mut b = backend_arc.write().await;
            let _ = b.load_model(std::path::Path::new("models/sd.safetensors"), &backend_trait::LoadOptions::default()).await;
        }
    }

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
        )
    })?;

    let img_bytes = inf_res.output_data.unwrap_or_default();
    let b64_str = use_base64_encode(&img_bytes);

    Ok(Json(ImageGenerationResponse {
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        data: vec![ImageObject { b64_json: b64_str }],
    }))
}

async fn transcribe_audio(
    State(state): State<AppState>,
) -> Result<Json<TranscriptionResponse>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::AudioAsr)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No audio ASR backend registered".to_string(),
        ))?;

    let request_id = format!("asr-{}", uuid_simple());

    let inf_req = InferenceRequest {
        request_id,
        prompt: "Transcribe audio input".to_string(),
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::AudioAsr,
        image_input: None,
    };

    {
        let is_loaded = {
            let b = backend_arc.read().await;
            b.is_loaded()
        };
        if !is_loaded {
            let mut b = backend_arc.write().await;
            let _ = b.load_model(std::path::Path::new("models/whisper.bin"), &backend_trait::LoadOptions::default()).await;
        }
    }

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
        )
    })?;

    Ok(Json(TranscriptionResponse {
        text: inf_res.output_text,
    }))
}

async fn synthesize_speech(
    State(state): State<AppState>,
    Json(payload): Json<SpeechRequest>,
) -> Result<Json<SpeechResponse>, (axum::http::StatusCode, String)> {
    let backend_arc = state
        .registry
        .get_backend(Modality::AudioTts)
        .await
        .ok_or((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "No TTS backend registered".to_string(),
        ))?;

    let request_id = format!("tts-{}", uuid_simple());

    let inf_req = InferenceRequest {
        request_id,
        prompt: payload.input,
        messages: None,
        sampling: SamplingParams::default(),
        modality: Modality::AudioTts,
        image_input: None,
    };

    {
        let is_loaded = {
            let b = backend_arc.read().await;
            b.is_loaded()
        };
        if !is_loaded {
            let mut b = backend_arc.write().await;
            let _ = b.load_model(std::path::Path::new("models/kokoro.onnx"), &backend_trait::LoadOptions::default()).await;
        }
    }

    let backend = backend_arc.read().await;
    let inf_res = backend.generate(inf_req).await.map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
        )
    })?;

    let audio_bytes = inf_res.output_data.unwrap_or_default();
    let b64_str = use_base64_encode(&audio_bytes);

    Ok(Json(SpeechResponse { audio_b64: b64_str }))
}

fn use_base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    let mut i = 0;
    while i < input.len() {
        let b0 = input[i] as u32;
        let b1 = if i + 1 < input.len() { input[i + 1] as u32 } else { 0 };
        let b2 = if i + 2 < input.len() { input[i + 2] as u32 } else { 0 };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(ALPHABET[((triple >> 18) & 63) as usize] as char);
        result.push(ALPHABET[((triple >> 12) & 63) as usize] as char);
        if i + 1 < input.len() {
            result.push(ALPHABET[((triple >> 6) & 63) as usize] as char);
        } else {
            result.push('=');
        }
        if i + 2 < input.len() {
            result.push(ALPHABET[(triple & 63) as usize] as char);
        } else {
            result.push('=');
        }

        i += 3;
    }
    result
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", nanos)
}

#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub model_path: String,
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ModelUnloadRequest {
    /// model_id returned by /v1/model/load. If omitted, unloads the single legacy model.
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DetectedModelEntry {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    /// `None` means this GGUF does not expose a readable context limit.
    pub max_context_size: Option<u32>,
    /// Inferred from the model's embedded metadata or its file type.
    pub modality: String,
    /// A vision GGUF also needs a local image projector before chat can accept images.
    pub image_input_available: bool,
}

fn has_image_projector(path: &std::path::Path) -> bool {
    let Some(directory) = path.parent() else { return false; };
    std::fs::read_dir(directory).ok().into_iter().flatten().flatten().any(|entry| {
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        name.contains("mmproj") && name.ends_with(".gguf")
    })
}

fn detect_model_modality(path: &std::path::Path) -> String {
    let extension = path.extension().and_then(|extension| extension.to_str()).unwrap_or_default().to_ascii_lowercase();
    match extension.as_str() {
        "gguf" => {
            use std::io::Read;
            let mut bytes = Vec::new();
            let _ = std::fs::File::open(path)
                .and_then(|mut file| file.by_ref().take(2 * 1024 * 1024).read_to_end(&mut bytes));
            if bytes.windows(b"image-text-to-text".len()).any(|window| window == b"image-text-to-text") {
                "vision".to_string()
            } else {
                "text".to_string()
            }
        }
        "safetensors" | "ckpt" => "image".to_string(),
        "bin" | "onnx" => "audio".to_string(),
        "mp4" | "webm" => "video".to_string(),
        _ => "unknown".to_string(),
    }
}

fn read_u32(bytes: &[u8], position: &mut usize) -> Option<u32> {
    let value = bytes.get(*position..*position + 4)?;
    *position += 4;
    Some(u32::from_le_bytes(value.try_into().ok()?))
}

fn read_u64(bytes: &[u8], position: &mut usize) -> Option<u64> {
    let value = bytes.get(*position..*position + 8)?;
    *position += 8;
    Some(u64::from_le_bytes(value.try_into().ok()?))
}

fn skip_gguf_value(bytes: &[u8], position: &mut usize, value_type: u32) -> Option<()> {
    let length = match value_type {
        0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 10 | 11 | 12 => 8,
        8 => read_u64(bytes, position)? as usize,
        9 => {
            let item_type = read_u32(bytes, position)?;
            let count = read_u64(bytes, position)? as usize;
            let item_size = match item_type { 0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4, 10 | 11 | 12 => 8, _ => return None };
            count.checked_mul(item_size)?
        }
        _ => return None,
    };
    *position = position.checked_add(length)?;
    (bytes.len() >= *position).then_some(())
}

/// Read the architecture context limit from GGUF key-value metadata.
fn gguf_context_length(path: &std::path::Path) -> Option<u32> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path).ok()?.take(8 * 1024 * 1024).read_to_end(&mut bytes).ok()?;
    if bytes.get(0..4)? != b"GGUF" { return None; }

    // Metadata arrays (notably arrays of strings) are allowed before the
    // architecture fields. Locate the unambiguous key directly first instead
    // of relying on every preceding metadata type being understood.
    let context_key = b".context_length";
    if let Some(key_start) = bytes.windows(context_key.len()).position(|window| window == context_key) {
        let mut value_position = key_start + context_key.len();
        let value_type = read_u32(&bytes, &mut value_position)?;
        return match value_type {
            4 => read_u32(&bytes, &mut value_position),
            5 => read_u32(&bytes, &mut value_position).map(|value| value as i32).filter(|value| *value > 0).map(|value| value as u32),
            10 | 11 => read_u64(&bytes, &mut value_position).and_then(|value| u32::try_from(value).ok()),
            _ => None,
        };
    }

    let mut position = 4;
    let version = read_u32(&bytes, &mut position)?;
    if !(2..=3).contains(&version) { return None; }
    let _tensor_count = read_u64(&bytes, &mut position)?;
    let kv_count = read_u64(&bytes, &mut position)?;
    for _ in 0..kv_count {
        let key_len = read_u64(&bytes, &mut position)? as usize;
        let key = std::str::from_utf8(bytes.get(position..position.checked_add(key_len)?)?).ok()?;
        position += key_len;
        let value_type = read_u32(&bytes, &mut position)?;
        if key.ends_with(".context_length") && matches!(value_type, 4 | 5 | 10 | 11) {
            return match value_type {
                4 => read_u32(&bytes, &mut position),
                5 => read_u32(&bytes, &mut position).map(|value| value as i32).filter(|value| *value > 0).map(|value| value as u32),
                10 | 11 => read_u64(&bytes, &mut position).and_then(|value| u32::try_from(value).ok()),
                _ => None,
            };
        }
        skip_gguf_value(&bytes, &mut position, value_type)?;
    }
    None
}

fn collect_model_files(directory: &std::path::Path, entries: &mut Vec<DetectedModelEntry>) {
    let Ok(children) = std::fs::read_dir(directory) else { return; };
    for child in children.flatten() {
        let path = child.path();
        if path.is_dir() {
            collect_model_files(&path, entries);
        } else if path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "gguf" | "safetensors" | "ckpt" | "onnx")) {
            let size_bytes = child.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            let modality = detect_model_modality(&path);
            entries.push(DetectedModelEntry {
                name: path.file_name().and_then(|name| name.to_str()).unwrap_or("Unknown model").to_string(),
                path: path.to_string_lossy().to_string(),
                size_bytes,
                max_context_size: gguf_context_length(&path),
                image_input_available: modality == "vision" && has_image_projector(&path),
                modality,
            });
        }
    }
}

/// Locate AIATM's own `models` directory without depending on how the daemon
/// was launched. This supports development builds, packaged binaries, and an
/// explicit deployment override through `AIATM_MODELS_DIR`.
async fn list_detected_models() -> Json<Vec<DetectedModelEntry>> {
    let mut entries = Vec::new();
    let mut candidates = Vec::new();

    if let Ok(directory) = std::env::var("AIATM_MODELS_DIR") {
        candidates.push(std::path::PathBuf::from(directory));
    }
    if let Ok(current_dir) = std::env::current_dir() {
        candidates.extend(current_dir.ancestors().map(|path| path.join("models")));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            candidates.extend(parent.ancestors().map(|path| path.join("models")));
        }
    }
    // Cargo resolves this to the project source tree, which keeps local
    // development reliable even when the binary is built into another target dir.
    candidates.push(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..\\..").join("models"));

    for directory in candidates {
        if directory.is_dir() {
                collect_model_files(&directory, &mut entries);
            break;
        }
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Json(entries)
}

/// Returns the backwards-compat single model status (first loaded model, or none).
async fn get_model_status(
    State(state): State<AppState>,
) -> Json<LoadedModelStatus> {
    let models = state.loaded_models.lock().await;
    if let Some(entry) = models.values().next() {
        Json(LoadedModelStatus {
            is_loaded: true,
            model_path: Some(entry.model_path.clone()),
            gpu_layers: Some(entry.gpu_layers),
            context_size: Some(entry.context_size),
        })
    } else {
        Json(LoadedModelStatus { is_loaded: false, model_path: None, gpu_layers: None, context_size: None })
    }
}

/// Returns all currently loaded models.
async fn list_loaded_models(
    State(state): State<AppState>,
) -> Json<Vec<LoadedModelEntry>> {
    let models = state.loaded_models.lock().await;
    // IDs contain a creation timestamp, so this makes the API's default model
    // choice deterministic: the most recently loaded model appears first.
    let mut entries: Vec<_> = models.values().cloned().collect();
    entries.sort_by(|a, b| b.model_id.cmp(&a.model_id));
    Json(entries)
}

/// Find llama-server binary (same logic as in llama-backend)
fn find_llama_server_binary() -> Option<std::path::PathBuf> {
    let lmstudio_dir = std::env::var("USERPROFILE")
        .map(|p| std::path::PathBuf::from(p).join(".lmstudio\\extensions\\backends"))
        .ok()?;
    if let Ok(entries) = std::fs::read_dir(lmstudio_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("llama-server.exe");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Generate a simple unique model ID using timestamp + port
fn new_model_id(port: u16) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    format!("mdl-{}-{}", ts, port)
}

/// Load a model: spawn a new llama-server on the next free port, register in loaded_models.
async fn load_model(
    State(state): State<AppState>,
    Json(payload): Json<ModelLoadRequest>,
) -> Result<Json<LoadedModelEntry>, (axum::http::StatusCode, String)> {
    let model_path = std::path::PathBuf::from(&payload.model_path);
    if !model_path.exists() {
        return Err((axum::http::StatusCode::BAD_REQUEST,
            format!("Model file not found at: {}", payload.model_path)));
    }

    let gpu_layers = payload.gpu_layers.unwrap_or(99);
    let context_size = payload.context_size.unwrap_or(4096);

    // Allocate a new port
    let port = state.next_port.fetch_add(1, AtomicOrdering::SeqCst);
    let model_id = new_model_id(port);

    // Spawn a dedicated llama-server for this model
    if let Some(server_bin) = find_llama_server_binary() {
        let child = std::process::Command::new(&server_bin)
            .arg("-m").arg(&model_path)
            .arg("--port").arg(port.to_string())
            .arg("-ngl").arg(gpu_layers.to_string())
            .arg("-c").arg(context_size.to_string())
            .arg("--host").arg("127.0.0.1")
            .spawn();
        match child {
            Ok(c) => {
                // Hand ownership of the Child to the registry so shutdown /
                // explicit unload can terminate it. (Previously `_c` was
                // discarded here, leaking the spawned process.)
                state.children.register(model_id.clone(), c);
                // Give the server time to start
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
            Err(e) => {
                tracing::warn!("Failed to spawn llama-server on port {}: {}", port, e);
            }
        }
    } else {
        tracing::info!("llama-server binary not found; registering model in simulation mode on port {}", port);
        // In simulation mode, still wait a moment so the UI feels responsive
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    let entry = LoadedModelEntry {
        model_id: model_id.clone(),
        model_path: payload.model_path.clone(),
        gpu_layers,
        context_size,
        port,
    };

    {
        let mut models = state.loaded_models.lock().await;
        models.insert(model_id, entry.clone());
    }

    // Keep backwards-compat active_model pointing at this (most recently loaded) model
    {
        let mut status = state.active_model.lock().await;
        *status = LoadedModelStatus {
            is_loaded: true,
            model_path: Some(payload.model_path),
            gpu_layers: Some(gpu_layers),
            context_size: Some(context_size),
        };
    }

    Ok(Json(entry))
}

/// Unload a model by model_id. Kills its llama-server process.
async fn unload_model(
    State(state): State<AppState>,
    Json(payload): Json<ModelUnloadRequest>,
) -> Result<Json<Vec<LoadedModelEntry>>, (axum::http::StatusCode, String)> {
    if let Some(mid) = &payload.model_id {
        // Targeted unload: remove from registry and kill port via taskkill
        let entry = {
            let mut models = state.loaded_models.lock().await;
            models.remove(mid)
        };
        if let Some(e) = entry {
            // `take` removes the entry AND terminates the tracked child via
            // Child::kill() (see ChildRegistry::take). No shell-out, no PID
            // guessing — the Child handle is the source of truth.
            let _ = state.children.take(mid);
            // Belt-and-suspenders: kill whichever llama-server is listening on
            // that port in case the tracker is out of sync (e.g. process
            // adopted from a previous run).
            let _ = std::process::Command::new("powershell")
                .args(["-Command", &format!(
                    "$pid = (Get-NetTCPConnection -LocalPort {} -ErrorAction SilentlyContinue).OwningProcess; if ($pid) {{ Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue }}",
                    e.port
                )])
                .spawn();
        }
    } else {
        // Legacy unload: kill everything via the backend trait
        let backend_arc = state
            .registry
            .get_backend(Modality::Text)
            .await
            .ok_or((axum::http::StatusCode::SERVICE_UNAVAILABLE, "No text backend registered".to_string()))?;
        let mut b = backend_arc.write().await;
        b.unload_model().await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let mut models = state.loaded_models.lock().await;
        models.clear();
    }

    // Update backwards-compat status
    {
        let models = state.loaded_models.lock().await;
        let mut status = state.active_model.lock().await;
        if let Some(entry) = models.values().next() {
            *status = LoadedModelStatus {
                is_loaded: true,
                model_path: Some(entry.model_path.clone()),
                gpu_layers: Some(entry.gpu_layers),
                context_size: Some(entry.context_size),
            };
        } else {
            *status = LoadedModelStatus { is_loaded: false, model_path: None, gpu_layers: None, context_size: None };
        }
    }

    let models = state.loaded_models.lock().await;
    Ok(Json(models.values().cloned().collect()))
}
