use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct MediaEntry {
    pub data: Vec<u8>,
    pub mime_type: String,
    pub created_at: Instant,
}

pub struct MediaStore {
    entries: RwLock<HashMap<String, MediaEntry>>,
    ttl: Duration,
}

impl MediaStore {
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }

    /// Store media data and return its ID (not the full URL path — the caller constructs that).
    pub async fn store(&self, data: Vec<u8>, mime_type: &str) -> String {
        let id = Uuid::new_v4().to_string();
        let entry = MediaEntry {
            data,
            mime_type: mime_type.to_string(),
            created_at: Instant::now(),
        };
        self.entries.write().await.insert(id.clone(), entry);
        id
    }

    /// Retrieve media by ID. Returns None if not found or expired.
    pub async fn get(&self, id: &str) -> Option<(Vec<u8>, String)> {
        let lock = self.entries.read().await;
        if let Some(entry) = lock.get(id) {
            if entry.created_at.elapsed() < self.ttl {
                return Some((entry.data.clone(), entry.mime_type.clone()));
            }
        }
        None
    }

    /// Remove expired entries. Call this periodically.
    pub async fn cleanup_expired(&self) {
        let mut lock = self.entries.write().await;
        lock.retain(|_, entry| entry.created_at.elapsed() < self.ttl);
    }
}
