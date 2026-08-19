use std::path::{Path, PathBuf};
use async_trait::async_trait;
use backend_trait::{
    BackendError, GenerationProgress, ImageParams, InferenceBackend, InferenceChunk, InferenceRequest,
    InferenceResponse, InferenceStream, LoadOptions, Modality, VramEstimate,
};
use futures::stream;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};
use base64::Engine;

/// Live progress reported by the `sd.exe` (GGUF) generation path, parsed
/// from the child process's stderr.
struct GgufProgress {
    active: bool,
    step: u32,
    total: u32,
    phase: String,
}

/// Ports handed out to spawned `diffusers_server.py` instances.
static NEXT_SD_PORT: AtomicU16 = AtomicU16::new(59200);

/// Claim the next port nothing is listening on. The counter restarts at 59200
/// on every daemon start, so without this probe a new server can collide with
/// an orphaned one left behind by a previous run.
fn next_free_sd_port() -> u16 {
    for _ in 0..64 {
        let port = NEXT_SD_PORT.fetch_add(1, Ordering::SeqCst);
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
        warn!("SD port {} already in use (orphaned server?), trying the next one", port);
    }
    NEXT_SD_PORT.fetch_add(1, Ordering::SeqCst)
}

/// How long to wait for `diffusers_server.py` to print "READY" on startup.
/// Diffusion pipelines (and a possible first-time HF download) can take a
/// while to load.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

pub struct SdBackend {
    model_path: Option<PathBuf>,
    is_loaded: Arc<AtomicBool>,
    supported_modalities: Vec<Modality>,
    /// Set when the loaded model is a `.gguf` file and `sd.exe` was found on
    /// disk. Presence of this field selects the one-shot CLI generation path.
    sd_binary: Option<PathBuf>,
    /// Set when the loaded model is a `.safetensors` file / HF-repo directory
    /// and `scripts/diffusers_server.py` was spawned successfully. Presence
    /// of this field selects the HTTP generation path.
    process: Option<Child>,
    port: u16,
    /// Live progress for the `sd.exe` (GGUF) generation path.
    gguf_progress: Arc<Mutex<GgufProgress>>,
}

impl SdBackend {
    pub fn new() -> Self {
        Self {
            model_path: None,
            is_loaded: Arc::new(AtomicBool::new(false)),
            supported_modalities: vec![Modality::Image],
            sd_binary: None,
            process: None,
            port: 0,
            gguf_progress: Arc::new(Mutex::new(GgufProgress {
                active: false,
                step: 0,
                total: 0,
                phase: String::new(),
            })),
        }
    }

    /// Locate the `stable-diffusion.cpp` CLI executable. Checked in order:
    /// an explicit `AIATM_SD_BINARY` env var override, known install
    /// locations (mirrors `llama-backend`'s `find_llama_server_binary`), then
    /// the system PATH.
    fn find_sd_binary() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("AIATM_SD_BINARY") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }

        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            let backends_dir = PathBuf::from(&userprofile).join(".lmstudio\\extensions\\backends");

            let direct = backends_dir.join("sd.exe");
            if direct.exists() {
                return Some(direct);
            }

            if let Ok(entries) = std::fs::read_dir(&backends_dir) {
                for entry in entries.flatten() {
                    let candidate = entry.path().join("sd.exe");
                    if candidate.exists() {
                        return Some(candidate);
                    }
                }
            }
        }

        #[cfg(windows)]
        {
            if let Ok(output) = Command::new("where").arg("sd.exe").output() {
                if output.status.success() {
                    if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                        let path = PathBuf::from(first_line.trim());
                        if path.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }
        #[cfg(not(windows))]
        {
            if let Ok(output) = Command::new("which").arg("sd").output() {
                if output.status.success() {
                    if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                        let path = PathBuf::from(first_line.trim());
                        if path.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }

        None
    }

    /// Spawn `scripts/diffusers_server.py` and block (on a worker thread)
    /// until it prints "READY" on stdout or `STARTUP_TIMEOUT` elapses.
    fn spawn_diffusers_server(model_path: &Path, port: u16) -> Result<Child, String> {
        let script_path = PathBuf::from("scripts").join("diffusers_server.py");
        if !script_path.exists() {
            return Err(format!(
                "diffusers_server.py script not found at: {}",
                script_path.display()
            ));
        }

        let mut cmd = Command::new("python");
        cmd.arg(&script_path)
            .arg("--model-path")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn diffusers_server.py: {}", e))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture stdout of diffusers_server.py".to_string())?;

        let start = Instant::now();
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    return Err("diffusers_server.py exited before signalling READY".to_string());
                }
                Ok(_) => {
                    if line.trim() == "READY" {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error reading diffusers_server.py stdout: {}", e));
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let _ = child.kill();
                return Err("Timed out waiting for diffusers_server.py to start".to_string());
            }
        }

        Ok(child)
    }

    /// Parse a whitespace-delimited `N/M` step token (e.g. from a
    /// `  |====>  | 3/20 - 1.50it/s` sd.exe progress line) out of a chunk of
    /// stderr text. Returns `Some((step, total))` for the last such token
    /// found in the chunk, ignoring tokens like `1.50it/s` that don't parse
    /// as two integers.
    fn parse_step_total(line: &str) -> Option<(u32, u32)> {
        let mut found = None;
        for tok in line.split_whitespace() {
            if let Some((a, b)) = tok.split_once('/') {
                if let (Ok(step), Ok(total)) = (a.parse::<u32>(), b.parse::<u32>()) {
                    if total > 0 {
                        found = Some((step, total));
                    }
                }
            }
        }
        found
    }

    /// Run `sd.exe` against a temp output PNG, tracking live progress via
    /// `gguf_progress`, and return the resulting PNG bytes.
    async fn generate_gguf(
        sd_binary: &Path,
        model_path: &Path,
        request_id: &str,
        prompt: &str,
        params: &ImageParams,
        gguf_progress: &Arc<Mutex<GgufProgress>>,
    ) -> Result<Vec<u8>, BackendError> {
        let out_path = std::env::temp_dir().join(format!("aiatm_sd_{}.png", request_id));
        let total_steps = params.steps.unwrap_or(20);

        let mut cmd = tokio::process::Command::new(sd_binary);
        cmd.arg("-m").arg(model_path).arg("-p").arg(prompt);
        if let Some(negative) = &params.negative_prompt {
            cmd.arg("-n").arg(negative);
        }
        cmd.arg("--steps").arg(total_steps.to_string());
        cmd.arg("--cfg-scale").arg(params.cfg_scale.unwrap_or(7.0).to_string());
        cmd.arg("-W").arg(params.width.unwrap_or(512).to_string());
        cmd.arg("-H").arg(params.height.unwrap_or(512).to_string());
        if let Some(seed) = params.seed {
            cmd.arg("-s").arg(seed.to_string());
        }
        cmd.arg("-o").arg(&out_path);
        cmd.stdout(Stdio::null()).stderr(Stdio::piped());

        {
            let mut progress = gguf_progress.lock().unwrap();
            progress.active = true;
            progress.step = 0;
            progress.total = total_steps;
            progress.phase = "generating".to_string();
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| BackendError::InferenceError(format!("Failed to run sd.exe: {}", e)))?;

        let stderr = child.stderr.take();
        let mut stderr_buf = Vec::new();

        let read_task = if let Some(stderr) = stderr {
            let progress = gguf_progress.clone();
            Some(tokio::spawn(async move {
                use tokio::io::AsyncReadExt;
                let mut reader = stderr;
                let mut buf = [0u8; 4096];
                let mut line_acc: Vec<u8> = Vec::new();
                let mut all = Vec::new();
                loop {
                    match reader.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => {
                            all.extend_from_slice(&buf[..n]);
                            for &b in &buf[..n] {
                                if b == b'\r' || b == b'\n' {
                                    if !line_acc.is_empty() {
                                        let line = String::from_utf8_lossy(&line_acc).to_string();
                                        if let Some((step, total)) = Self::parse_step_total(&line) {
                                            let mut p = progress.lock().unwrap();
                                            p.step = step;
                                            p.total = total;
                                        }
                                        line_acc.clear();
                                    }
                                } else {
                                    line_acc.push(b);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                if !line_acc.is_empty() {
                    let line = String::from_utf8_lossy(&line_acc).to_string();
                    if let Some((step, total)) = Self::parse_step_total(&line) {
                        let mut p = progress.lock().unwrap();
                        p.step = step;
                        p.total = total;
                    }
                }
                all
            }))
        } else {
            None
        };

        let status = child
            .wait()
            .await
            .map_err(|e| BackendError::InferenceError(format!("Failed to wait on sd.exe: {}", e)))?;

        if let Some(task) = read_task {
            if let Ok(bytes) = task.await {
                stderr_buf = bytes;
            }
        }

        {
            let mut progress = gguf_progress.lock().unwrap();
            progress.active = false;
        }

        if !status.success() {
            let stderr = String::from_utf8_lossy(&stderr_buf).to_string();
            let _ = tokio::fs::remove_file(&out_path).await;
            return Err(BackendError::InferenceError(format!(
                "sd.exe exited with status {:?}: {}",
                status.code(),
                stderr
            )));
        }

        let bytes = tokio::fs::read(&out_path).await.map_err(|e| {
            BackendError::InferenceError(format!("Failed to read sd.exe output PNG: {}", e))
        })?;
        let _ = tokio::fs::remove_file(&out_path).await;
        Ok(bytes)
    }

    /// POST the prompt+params to the running `diffusers_server.py` instance
    /// and decode the returned base64 PNG.
    async fn generate_diffusers(
        port: u16,
        prompt: &str,
        params: &ImageParams,
    ) -> Result<Vec<u8>, BackendError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let payload = serde_json::json!({
            "prompt": prompt,
            "negative_prompt": params.negative_prompt,
            "steps": params.steps.unwrap_or(20),
            "guidance_scale": params.cfg_scale.unwrap_or(7.5),
            "width": params.width.unwrap_or(512),
            "height": params.height.unwrap_or(512),
            "seed": params.seed,
        });

        let url = format!("http://127.0.0.1:{}/generate", port);
        let response = client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| BackendError::InferenceError(format!("diffusers_server connection error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let err_text = response.text().await.unwrap_or_default();
            return Err(BackendError::InferenceError(format!(
                "diffusers_server HTTP {} error: {}",
                status, err_text
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| BackendError::InferenceError(format!("diffusers_server response parse error: {}", e)))?;

        let image_b64 = body["image_base64"]
            .as_str()
            .ok_or_else(|| BackendError::InferenceError("diffusers_server response missing image_base64".to_string()))?;

        base64::engine::general_purpose::STANDARD
            .decode(image_b64)
            .map_err(|e| BackendError::InferenceError(format!("Failed to decode image_base64: {}", e)))
    }

    /// GET `/progress` from the running `diffusers_server.py` instance and
    /// map its JSON into a `GenerationProgress`. `job_id`/`modality` are left
    /// for daemon-core to stamp.
    async fn poll_diffusers_progress(port: u16) -> Option<GenerationProgress> {
        let client = reqwest::Client::builder().no_proxy().build().ok()?;
        let url = format!("http://127.0.0.1:{}/progress", port);
        let response = client.get(&url).send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        let body: serde_json::Value = response.json().await.ok()?;
        if body.get("status").and_then(|v| v.as_str()) == Some("idle") {
            return None;
        }

        let updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Some(GenerationProgress {
            job_id: String::new(),
            modality: "image".to_string(),
            phase: body.get("phase").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            step: body.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            total: body.get("total").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            percent: body.get("percent").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            status: body.get("status").and_then(|v| v.as_str()).unwrap_or("running").to_string(),
            message: body.get("message").and_then(|v| v.as_str()).map(|s| s.to_string()),
            updated_at,
        })
    }
}

impl Default for SdBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SdBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[async_trait]
impl InferenceBackend for SdBackend {
    fn name(&self) -> &'static str {
        "stable-diffusion"
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
        // Kill any previously running diffusers server and clear the sd.exe
        // binary reference before deciding on the new model's mode.
        if let Some(mut child) = self.process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.sd_binary = None;

        if !model_path.exists() {
            info!(
                "SD model path '{}' not found on disk; starting backend in simulation mode.",
                model_path.display()
            );
            self.model_path = Some(model_path.to_path_buf());
            self.is_loaded.store(true, Ordering::SeqCst);
            return Ok(());
        }

        let is_gguf = model_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

        if is_gguf {
            info!("Loading GGUF Stable Diffusion model: {}", model_path.display());
            match Self::find_sd_binary() {
                Some(bin) => {
                    info!("Found stable-diffusion.cpp binary at: {}", bin.display());
                    self.sd_binary = Some(bin);
                }
                None => {
                    warn!(
                        "sd.exe binary not found (set AIATM_SD_BINARY or place it under \
                         ~/.lmstudio/extensions/backends/); falling back to simulation mode."
                    );
                }
            }
        } else {
            info!("Loading safetensors/HF-repo Stable Diffusion model: {}", model_path.display());
            let port = next_free_sd_port();
            let model_path_owned = model_path.to_path_buf();

            match tokio::task::spawn_blocking(move || Self::spawn_diffusers_server(&model_path_owned, port))
                .await
                .map_err(|e| BackendError::LoadError(format!("spawn_blocking join error: {}", e)))?
            {
                Ok(child) => {
                    info!("diffusers_server.py ready on port {} (PID: {})", port, child.id());
                    self.process = Some(child);
                    self.port = port;
                }
                Err(e) => {
                    warn!(
                        "Could not start the diffusers server for '{}': {}. Falling back to simulation mode.",
                        model_path.display(),
                        e
                    );
                }
            }
        }

        self.model_path = Some(model_path.to_path_buf());
        self.is_loaded.store(true, Ordering::SeqCst);

        info!("Successfully loaded model in Stable Diffusion backend");
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<(), BackendError> {
        if self.is_loaded.load(Ordering::SeqCst) {
            info!("Unloading Stable Diffusion model from memory");
            if let Some(mut child) = self.process.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
            self.sd_binary = None;
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
        let params = request.image_params.clone().unwrap_or_default();

        info!("Executing image generation prompt: '{}'", request.prompt);

        let png_bytes: Vec<u8> = if let Some(sd_binary) = &self.sd_binary {
            let model_path = self.model_path.as_deref().ok_or(BackendError::ModelNotLoaded)?;
            Self::generate_gguf(
                sd_binary,
                model_path,
                &request.request_id,
                &request.prompt,
                &params,
                &self.gguf_progress,
            )
            .await?
        } else if self.process.is_some() {
            Self::generate_diffusers(self.port, &request.prompt, &params).await?
        } else {
            // Simulation mode: no sd.exe binary and no diffusers server running
            // (e.g. missing model file or missing runtime dependencies).
            vec![
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
                0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
                0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
                0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
                0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
                0x42, 0x60, 0x82,
            ]
        };

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

    async fn poll_progress(&self) -> Option<GenerationProgress> {
        if self.sd_binary.is_some() {
            let progress = self.gguf_progress.lock().unwrap();
            if !progress.active {
                return None;
            }
            let percent = if progress.total > 0 {
                progress.step as f32 / progress.total as f32 * 100.0
            } else {
                -1.0
            };
            let updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            return Some(GenerationProgress {
                job_id: String::new(),
                modality: String::new(),
                phase: progress.phase.clone(),
                step: progress.step,
                total: progress.total,
                percent,
                status: "running".to_string(),
                message: None,
                updated_at,
            });
        }

        if self.process.is_none() {
            // No diffusers_server.py running (simulation mode) — no progress
            // source available.
            return None;
        }
        Self::poll_diffusers_progress(self.port).await
    }
}
