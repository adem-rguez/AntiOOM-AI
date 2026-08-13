use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

#[derive(Debug, Clone)]
pub struct ExpertActivationTrace {
    pub layer_id: u32,
    pub expert_ids: Vec<u32>,
    pub timestamp: u64,
}

pub struct MoeExpertCache {
    max_cached_experts_per_layer: usize,
    activation_history: Arc<Mutex<VecDeque<ExpertActivationTrace>>>,
    hits: Arc<Mutex<u64>>,
    misses: Arc<Mutex<u64>>,
}

impl MoeExpertCache {
    pub fn new(max_cached_experts_per_layer: usize) -> Self {
        Self {
            max_cached_experts_per_layer,
            activation_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn record_activation(&self, layer_id: u32, expert_ids: Vec<u32>) {
        let mut history = self.activation_history.lock().await;
        if history.len() >= 1000 {
            history.pop_front();
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        history.push_back(ExpertActivationTrace {
            layer_id,
            expert_ids,
            timestamp: now,
        });
    }

    pub async fn predict_next_experts(&self, layer_id: u32) -> Vec<u32> {
        let history = self.activation_history.lock().await;
        let mut frequency_map: HashMap<u32, usize> = HashMap::new();

        for trace in history.iter().rev().take(100) {
            if trace.layer_id == layer_id {
                for &expert_id in &trace.expert_ids {
                    *frequency_map.entry(expert_id).or_insert(0) += 1;
                }
            }
        }

        let mut sorted: Vec<(u32, usize)> = frequency_map.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        sorted
            .into_iter()
            .take(self.max_cached_experts_per_layer)
            .map(|(expert_id, _)| expert_id)
            .collect()
    }

    pub async fn get_cache_stats(&self) -> (u64, u64, f64) {
        let hits = *self.hits.lock().await;
        let misses = *self.misses.lock().await;
        let total = hits + misses;
        let hit_rate = if total == 0 { 0.0 } else { (hits as f64) / (total as f64) };
        (hits, misses, hit_rate)
    }
}
