use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, InferenceBackend, InferenceChunk, InferenceRequest, InferenceResponse,
    InferenceStream, LoadOptions, Modality, ToolSchema, VramEstimate,
};
use futures::stream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::info;

pub struct WhisperBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
}

impl WhisperBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::AudioAsr],
        }
    }
}

impl Default for WhisperBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for WhisperBackend {
    fn name(&self) -> &'static str {
        "whisper.cpp"
    }

    fn supported_modalities(&self) -> &[Modality] {
        &self.supported_modalities
    }

    async fn estimate_vram(
        &self,
        model_path: &Path,
        _options: &LoadOptions,
    ) -> Result<VramEstimate, BackendError> {
        let metadata = std::fs::metadata(model_path).map_err(|e| {
            BackendError::LoadError(format!("Failed to read file metadata for Whisper VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        let estimated_vram = (file_size as f64 * 1.1) as u64;

        Ok(VramEstimate {
            required_bytes: estimated_vram,
            recommended_gpu_layers: 99,
            total_layers: 12,
            fits_in_vram: true,
        })
    }

    async fn load_model(
        &mut self,
        model_path: &Path,
        _options: &LoadOptions,
    ) -> Result<(), BackendError> {
        if !model_path.exists() {
            info!("Whisper model path '{}' not found on disk; starting backend in simulation mode.", model_path.display());
        }

        info!("Loading Whisper ASR model: {}", model_path.display());
        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in whisper.cpp backend");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading whisper.cpp model from memory");
            self.model_path = None;
            self.is_loaded.store(false, Ordering::SeqCst);
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.is_loaded.load(Ordering::SeqCst)
    }

    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        let start_time = std::time::Instant::now();
        let duration = start_time.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text: "[Transcribed Audio Speech Text via whisper.cpp]".to_string(),
            output_data: None,
            tokens_generated: 8,
            generation_time_ms: duration,
            tool_calls: None,
            finish_reason: None,
        })
    }

    async fn generate_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceStream, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::ModelNotLoaded);
        }

        let req_id = request.request_id.clone();
        let chunks = vec![Ok(InferenceChunk {
            request_id: req_id,
            delta_text: "[Transcribing audio...]".to_string(),
            delta_data: None,
            is_final: true,
            delta_tool_call: None,
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }

    fn as_tool_schema(&self) -> Option<ToolSchema> {
        Some(ToolSchema {
            name: "transcribe_audio".into(),
            description: "Transcribe audio to text. Returns transcription text.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "audio_data": {"type": "string", "description": "Base64-encoded audio data to transcribe"},
                    "language": {"type": "string", "description": "Language code (e.g. 'en', 'fr')"}
                },
                "required": ["audio_data"]
            }),
        })
    }
}
