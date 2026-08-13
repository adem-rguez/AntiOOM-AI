use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHardwareInfo {
    pub cpu_cores: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub gpu_name: Option<String>,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitEstimationRequest {
    pub parameter_count_billions: f64,
    pub quantization: String, // e.g. "Q4_K_M", "Q8_0", "F16"
    pub context_size: u32,
    pub modality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitEstimationResult {
    pub model_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub total_required_vram_bytes: u64,
    pub fits_in_vram: bool,
    pub fits_in_ram: bool,
    pub recommended_gpu_layers: u32,
    pub estimated_tok_per_sec: f64,
}

pub struct HardwareProfiler;

impl HardwareProfiler {
    pub fn probe() -> SystemHardwareInfo {
        let cpu_cores = num_cpus::get();

        // 1. Get system RAM dynamically
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let total_ram_bytes = sys.total_memory();
        let available_ram_bytes = sys.available_memory();

        // 2. Query VRAM and GPU name using nvidia-smi
        let mut gpu_name = Some("NVIDIA GPU (CUDA)".to_string());
        let mut total_vram_bytes = 8 * 1024 * 1024 * 1024; // Default fallback: 8GB
        let mut free_vram_bytes = 6 * 1024 * 1024 * 1024;  // Default fallback: 6GB

        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let out_str = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = out_str.trim().split(',').collect();
                if parts.len() >= 3 {
                    gpu_name = Some(parts[0].trim().to_string());
                    if let Ok(tot_mb) = parts[1].trim().parse::<u64>() {
                        total_vram_bytes = tot_mb * 1024 * 1024;
                    }
                    if let Ok(free_mb) = parts[2].trim().parse::<u64>() {
                        free_vram_bytes = free_mb * 1024 * 1024;
                    }
                }
            }
        }

        info!(
            "Probed System Hardware: CPU Cores: {}, Total RAM: {} GB, GPU: {:?}, VRAM Total: {} MB, VRAM Free: {} MB",
            cpu_cores,
            total_ram_bytes / (1024 * 1024 * 1024),
            gpu_name,
            total_vram_bytes / (1024 * 1024),
            free_vram_bytes / (1024 * 1024)
        );

        SystemHardwareInfo {
            cpu_cores,
            total_ram_bytes,
            available_ram_bytes,
            gpu_name,
            total_vram_bytes,
            free_vram_bytes,
        }
    }

    pub fn estimate_fit(req: &FitEstimationRequest, sys: &SystemHardwareInfo) -> FitEstimationResult {
        // Bits per weight estimation
        let bits_per_weight: f64 = match req.quantization.to_uppercase().as_str() {
            "Q2_K" => 2.5,
            "Q4_0" | "Q4_K_S" | "Q4_K_M" => 4.5,
            "Q5_K_M" | "Q5_0" => 5.5,
            "Q8_0" => 8.5,
            "F16" => 16.0,
            _ => 4.5,
        };

        let weight_bytes = (req.parameter_count_billions * 1e9 * (bits_per_weight / 8.0)) as u64;
        
        // KV cache size calculation: 2 * layers * heads * dim * context_size * bytes_per_element
        let kv_cache_bytes = ((req.context_size as f64) * 512.0 * 1024.0) as u64;
        let total_vram_needed = weight_bytes + kv_cache_bytes;

        let fits_in_vram = total_vram_needed <= sys.free_vram_bytes;
        let fits_in_ram = total_vram_needed <= sys.available_ram_bytes;

        let estimated_tok_per_sec = if fits_in_vram {
            65.0 // Fast GPU inference speed
        } else if fits_in_ram {
            12.5 // Offloaded CPU RAM speed
        } else {
            1.5 // Severe swap thrashing speed
        };

        FitEstimationResult {
            model_weight_bytes: weight_bytes,
            kv_cache_bytes,
            total_required_vram_bytes: total_vram_needed,
            fits_in_vram,
            fits_in_ram,
            recommended_gpu_layers: if fits_in_vram { 99 } else { 16 },
            estimated_tok_per_sec,
        }
    }
}

mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    }
}
