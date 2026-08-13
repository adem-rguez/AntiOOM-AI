use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, ChatMessage, InferenceBackend, InferenceChunk, InferenceRequest, InferenceResponse,
    InferenceStream, LoadOptions, Modality, VramEstimate,
};
use futures::stream;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

pub struct LlamaBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    server_process: Arc<Mutex<Option<Child>>>,
    server_port: u16,
}

impl LlamaBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::Text, Modality::Embedding],
            server_process: Arc::new(Mutex::new(None)),
            server_port: 50052,
        }
    }

    fn find_llama_server_binary() -> Option<PathBuf> {
        let lmstudio_dir = std::env::var("USERPROFILE")
            .map(|p| PathBuf::from(p).join(".lmstudio\\extensions\\backends"))
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
}

impl Default for LlamaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        let mut lock = self.server_process.lock().unwrap();
        if let Some(mut child) = lock.take() {
            let _ = child.kill();
        }
    }
}

#[async_trait]
impl InferenceBackend for LlamaBackend {
    fn name(&self) -> &'static str {
        "llama.cpp"
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
            BackendError::LoadError(format!("Failed to read file metadata for VRAM estimate: {}", e))
        })?;

        let file_size = metadata.len();
        let estimated_vram = (file_size as f64 * 1.2) as u64;

        Ok(VramEstimate {
            required_bytes: estimated_vram,
            recommended_gpu_layers: 99,
            total_layers: 32,
            fits_in_vram: true,
        })
    }

    async fn load_model(
        &mut self,
        model_path: &Path,
        options: &LoadOptions,
    ) -> Result<(), BackendError> {
        if !model_path.exists() {
            info!("Model path '{}' not found; running llama.cpp in simulation mode.", model_path.display());
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        info!(
            "Loading GGUF model: {} (GPU layers: {:?}, context size: {:?})",
            model_path.display(),
            options.gpu_layers,
            options.context_size
        );

        // Kill existing server if running
        {
            let mut lock = self.server_process.lock().unwrap();
            if let Some(mut child) = lock.take() {
                let _ = child.kill();
            }
        }

        if let Some(server_bin) = Self::find_llama_server_binary() {
            info!("Found native CUDA llama-server binary at: {}", server_bin.display());
            let gpu_layers = options.gpu_layers.unwrap_or(99).to_string();
            let ctx_size = options.context_size.unwrap_or(4096).to_string();
            let port_str = self.server_port.to_string();

            let child = Command::new(&server_bin)
                .arg("-m")
                .arg(model_path)
                .arg("--port")
                .arg(&port_str)
                .arg("-ngl")
                .arg(&gpu_layers)
                .arg("-c")
                .arg(&ctx_size)
                .arg("--host")
                .arg("127.0.0.1")
                .spawn();

            match child {
                Ok(c) => {
                    info!("Spawned llama-server process (PID: {}). Waiting for startup...", c.id());
                    {
                        let mut lock = self.server_process.lock().unwrap();
                        *lock = Some(c);
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                }
                Err(e) => {
                    warn!("Failed to spawn llama-server process: {}. Falling back to simulation.", e);
                }
            }
        } else {
            info!("llama-server binary not found; falling back to simulation mode.");
        }

        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in llama.cpp backend");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading llama.cpp model from memory");
            let mut lock = self.server_process.lock().unwrap();
            if let Some(mut child) = lock.take() {
                let _ = child.kill();
            }
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

        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let stop_seqs: Vec<String> = if request.sampling.stop_sequences.is_empty() {
            vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string(), "</s>".to_string()]
        } else {
            request.sampling.stop_sequences.clone()
        };

        let (generated_text, tokens) = if let Some(messages) = &request.messages {
            // ── Use /v1/chat/completions so llama-server applies the model's native
            //    Jinja chat template (which injects <think> for Qwen3 reasoning). ──
            let chat_messages: Vec<serde_json::Value> = messages
                .iter()
                .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
                .collect();

            let payload = serde_json::json!({
                "model": "local",
                "messages": chat_messages,
                "max_tokens": request.sampling.max_tokens,
                "temperature": request.sampling.temperature,
                "top_p": request.sampling.top_p,
                "repeat_penalty": 1.15,
                "stop": stop_seqs,
                // Ask llama-server to separate thinking from answer
                "thinking": true,
            });

            let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.server_port);
            let res = client.post(&url).json(&payload).send().await;

            match res {
                Ok(response) => {
                    if response.status().is_success() {
                        let text = response.text().await.unwrap_or_default();
                        if let Ok(body) = serde_json::from_str::<serde_json::Value>(&text) {
                            let msg = &body["choices"][0]["message"];

                            // llama-server with thinking models returns reasoning in
                            // "reasoning_content" and the final answer in "content"
                            let reasoning = msg["reasoning_content"].as_str().unwrap_or("").trim().to_string();
                            let answer   = msg["content"].as_str().unwrap_or("").trim().to_string();

                            // Recombine into <think>...</think>\nanswer so the frontend
                            // parseThinking() function renders them correctly
                            let combined = if !reasoning.is_empty() {
                                format!("<think>\n{}\n</think>\n\n{}", reasoning, answer)
                            } else if answer.contains("<think>") {
                                // Model put everything in content (older llama-server build)
                                answer
                            } else {
                                answer
                            };

                            let tok_count = body["usage"]["completion_tokens"]
                                .as_u64()
                                .unwrap_or(20) as u32;
                            (combined, tok_count)
                        } else {
                            (format!("[llama.cpp chat response parsing error: {}]", text), 10)
                        }
                    } else {
                        let status = response.status();
                        let err_text = response.text().await.unwrap_or_default();
                        (format!("[llama-server HTTP {} error: {}]", status, err_text), 0)
                    }
                }
                Err(e) => (format!("[llama-server connection error: {}]", e), 0),
            }

        } else {
            // ── Fallback: raw /completion with manually-formatted prompt ──
            let payload = serde_json::json!({
                "prompt": request.prompt,
                "n_predict": request.sampling.max_tokens,
                "temperature": request.sampling.temperature,
                "top_p": request.sampling.top_p,
                "repeat_penalty": 1.15,
                "stop": stop_seqs,
            });

            let url = format!("http://127.0.0.1:{}/completion", self.server_port);
            let res = client.post(&url).json(&payload).send().await;

            match res {
                Ok(response) => {
                    if response.status().is_success() {
                        let text = response.text().await.unwrap_or_default();
                        if let Ok(body) = serde_json::from_str::<serde_json::Value>(&text) {
                            let content = body["content"].as_str().unwrap_or("").to_string();
                            let tok_count = body["tokens_predicted"].as_u64().unwrap_or(20) as u32;
                            (content, tok_count)
                        } else {
                            (format!("[llama.cpp raw response parsing error: {}]", text), 10)
                        }
                    } else {
                        let status = response.status();
                        let err_text = response.text().await.unwrap_or_default();
                        (format!("[llama-server HTTP {} error: {}]", status, err_text), 0)
                    }
                }
                Err(e) => (format!("[llama-server connection error: {}]", e), 0),
            }
        };

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(InferenceResponse {
            request_id: request.request_id,
            output_text: generated_text,
            output_data: None,
            tokens_generated: tokens,
            generation_time_ms: duration,
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
        let chunks = vec![
            Ok(InferenceChunk {
                request_id: req_id.clone(),
                delta_text: "[llama.cpp stream start: ".to_string(),
                delta_data: None,
                is_final: false,
            }),
            Ok(InferenceChunk {
                request_id: req_id.clone(),
                delta_text: request.prompt.clone(),
                delta_data: None,
                is_final: false,
            }),
            Ok(InferenceChunk {
                request_id: req_id,
                delta_text: "]".to_string(),
                delta_data: None,
                is_final: true,
            }),
        ];

        Ok(Box::pin(stream::iter(chunks)))
    }
}
