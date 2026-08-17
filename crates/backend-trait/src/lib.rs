use std::path::Path;
use std::pin::Pin;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use futures::Stream;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    AudioAsr,
    AudioTts,
    Embedding,
    Video,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramEstimate {
    pub required_bytes: u64,
    pub recommended_gpu_layers: u32,
    pub total_layers: u32,
    pub fits_in_vram: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadOptions {
    pub gpu_layers: Option<u32>,
    pub context_size: Option<u32>,
    pub batch_size: Option<u32>,
    pub threads: Option<u32>,
    pub params: std::collections::HashMap<String, String>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            gpu_layers: Some(99), // Default: offload all layers if possible
            context_size: Some(4096),
            batch_size: Some(512),
            threads: Some(8),
            params: std::collections::HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 512,
            stop_sequences: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments_delta: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_handle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_data: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt: String,
    /// Optional structured chat messages. When provided, backends that support
    /// /v1/chat/completions (e.g. llama-server) will use these directly with
    /// the model's native chat template (enables Qwen3 <think> reasoning).
    pub messages: Option<Vec<ChatMessage>>,
    pub sampling: SamplingParams,
    pub modality: Modality,
    pub image_input: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub output_text: String,
    pub output_data: Option<Vec<u8>>,
    pub tokens_generated: u32,
    pub generation_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceChunk {
    pub request_id: String,
    pub delta_text: String,
    pub delta_data: Option<Vec<u8>>,
    pub is_final: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_tool_call: Option<ToolCallDelta>,
}

pub type InferenceStream = Pin<Box<dyn Stream<Item = Result<InferenceChunk, BackendError>> + Send>>;

#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Inference execution error: {0}")]
    InferenceError(String),
    #[error("Insufficient VRAM: required {required_bytes} B, available {available_bytes} B")]
    OutofVram { required_bytes: u64, available_bytes: u64 },
    #[error("Unsupported modality: {0:?}")]
    UnsupportedModality(Modality),
    #[error("Backend dynamic error: {0}")]
    Other(String),
}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Friendly name of this backend (e.g. "llama.cpp", "stable-diffusion.cpp")
    fn name(&self) -> &'static str;

    /// Supported model modalities
    fn supported_modalities(&self) -> &[Modality];

    /// Estimate VRAM usage for a given model file and load options
    async fn estimate_vram(
        &self,
        model_path: &Path,
        options: &LoadOptions,
    ) -> Result<VramEstimate, BackendError>;

    /// Load model into memory/GPU
    async fn load_model(
        &mut self,
        model_path: &Path,
        options: &LoadOptions,
    ) -> Result<(), BackendError>;

    /// Unload model from memory/GPU
    async fn unload_model(&mut self) -> Result<(), BackendError>;

    /// Check if a model is currently loaded in this backend
    fn is_loaded(&self) -> bool;

    /// Run full synchronous inference
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, BackendError>;

    /// Run streaming inference
    async fn generate_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, BackendError>;

    fn as_tool_schema(&self) -> Option<ToolSchema> { None }
}
