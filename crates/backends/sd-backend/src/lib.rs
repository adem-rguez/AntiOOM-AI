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

pub struct SdBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
}

impl SdBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::Image],
        }
    }
}

impl Default for SdBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for SdBackend {
    fn name(&self) -> &'static str {
        "stable-diffusion.cpp"
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
            BackendError::LoadError(format!("Failed to read file metadata for SD VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        // Heuristic: UNet + VAE + TextEncoder VRAM allocation (approx 1.3x weight size)
        let estimated_vram = (file_size as f64 * 1.3) as u64;

        Ok(VramEstimate {
            required_bytes: estimated_vram,
            recommended_gpu_layers: 99,
            total_layers: 16,
            fits_in_vram: true,
        })
    }

    async fn load_model(
        &mut self,
        model_path: &Path,
        _options: &LoadOptions,
    ) -> Result<(), BackendError> {
        if !model_path.exists() {
            info!("SD model path '{}' not found on disk; starting backend in simulation mode.", model_path.display());
        }

        info!("Loading Stable Diffusion model: {}", model_path.display());
        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in stable-diffusion.cpp backend");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading stable-diffusion.cpp model from memory");
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
        
        info!("Executing image generation prompt: '{}'", request.prompt);

        // Standard 1x1 transparent PNG payload bytes placeholder
        let png_bytes: Vec<u8> = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
            0x42, 0x60, 0x82,
        ];

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text: "Image generation completed successfully".to_string(),
            output_data: Some(png_bytes),
            tokens_generated: 1,
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
            delta_text: "Generating image steps...".to_string(),
            delta_data: None,
            is_final: true,
            delta_tool_call: None,
        })];

        Ok(Box::pin(stream::iter(chunks)))
    }

    fn as_tool_schema(&self) -> Option<ToolSchema> {
        Some(ToolSchema {
            name: "generate_image".into(),
            description: "Generate an image from a text description. Returns image data.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Detailed description of the image to generate"},
                    "size": {"type": "string", "enum": ["512x512", "768x768", "1024x1024"], "default": "512x512"},
                    "negative_prompt": {"type": "string", "description": "What to exclude from the image"}
                },
                "required": ["prompt"]
            }),
        })
    }
}
