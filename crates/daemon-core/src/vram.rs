use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct VramAllocation {
    pub model_id: String,
    pub backend_name: String,
    pub allocated_bytes: u64,
    pub last_used_timestamp: u64,
}

pub struct VramArbiter {
    total_capacity_bytes: u64,
    allocations: Arc<Mutex<Vec<VramAllocation>>>,
}

impl VramArbiter {
    pub fn new(total_capacity_bytes: u64) -> Self {
        Self {
            total_capacity_bytes,
            allocations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn get_allocated_bytes(&self) -> u64 {
        let allocs = self.allocations.lock().await;
        allocs.iter().map(|a| a.allocated_bytes).sum()
    }

    pub async fn get_free_bytes(&self) -> u64 {
        let allocated = self.get_allocated_bytes().await;
        self.total_capacity_bytes.saturating_sub(allocated)
    }

    pub async fn register_allocation(
        &self,
        model_id: String,
        backend_name: String,
        bytes: u64,
    ) -> bool {
        let mut allocs = self.allocations.lock().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        allocs.push(VramAllocation {
            model_id: model_id.clone(),
            backend_name,
            allocated_bytes: bytes,
            last_used_timestamp: now,
        });

        info!(
            "Registered VRAM allocation for model '{}': {} MB (Total used: {} MB / {} MB)",
            model_id,
            bytes / (1024 * 1024),
            allocs.iter().map(|a| a.allocated_bytes).sum::<u64>() / (1024 * 1024),
            self.total_capacity_bytes / (1024 * 1024)
        );

        true
    }

    pub async fn remove_allocation(&self, model_id: &str) {
        let mut allocs = self.allocations.lock().await;
        allocs.retain(|a| a.model_id != model_id);
        info!("Freed VRAM allocation for model '{}'", model_id);
    }
}
