use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerNode {
    pub node_id: String,
    pub address: String,
    pub free_vram_bytes: u64,
    pub total_vram_bytes: u64,
    pub latency_ms: u32,
}

pub struct ClusterPoolManager {
    local_node_id: String,
    nodes: Arc<RwLock<HashMap<String, PeerNode>>>,
}

impl ClusterPoolManager {
    pub fn new(local_node_id: String) -> Self {
        Self {
            local_node_id,
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_peer(&self, peer: PeerNode) {
        let mut lock = self.nodes.write().await;
        info!(
            "Registered LAN Peer Node '{}' ({}) - Free VRAM: {} MB",
            peer.node_id,
            peer.address,
            peer.free_vram_bytes / (1024 * 1024)
        );
        lock.insert(peer.node_id.clone(), peer);
    }

    pub async fn list_nodes(&self) -> Vec<PeerNode> {
        let lock = self.nodes.read().await;
        lock.values().cloned().collect()
    }
}
