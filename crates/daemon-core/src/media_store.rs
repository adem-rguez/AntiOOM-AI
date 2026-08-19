use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct MediaEntry {
    pub data: Vec<u8>,
    pub mime_type: String,
    pub created_at: Instant,
}

enum RetentionMode {
    Memory(Duration),
    Disk,
}

pub struct MediaStore {
    entries: RwLock<HashMap<String, MediaEntry>>,
    retention: std::sync::RwLock<RetentionMode>,
    disk_dir: PathBuf,
}

impl MediaStore {
    pub fn new(ttl_seconds: u64, disk_dir: PathBuf) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            retention: std::sync::RwLock::new(RetentionMode::Memory(Duration::from_secs(ttl_seconds))),
            disk_dir,
        }
    }

    /// Update retention behavior at runtime. `ttl_seconds` is ignored when `persist_disk` is true.
    pub fn set_retention(&self, ttl_seconds: u64, persist_disk: bool) {
        let mode = if persist_disk {
            if let Err(e) = std::fs::create_dir_all(&self.disk_dir) {
                tracing::warn!("Failed to create media disk_dir {:?}: {}", self.disk_dir, e);
            }
            RetentionMode::Disk
        } else {
            RetentionMode::Memory(Duration::from_secs(ttl_seconds))
        };
        *self.retention.write().unwrap() = mode;
    }

    /// Returns (ttl_seconds, persist_disk). In Disk mode ttl_seconds is 0.
    pub fn retention(&self) -> (u64, bool) {
        match &*self.retention.read().unwrap() {
            RetentionMode::Memory(ttl) => (ttl.as_secs(), false),
            RetentionMode::Disk => (0, true),
        }
    }

    /// Store media data and return its ID (not the full URL path — the caller constructs that).
    pub async fn store(&self, data: Vec<u8>, mime_type: &str) -> String {
        let id = Uuid::new_v4().to_string();
        let persist_disk = matches!(&*self.retention.read().unwrap(), RetentionMode::Disk);

        if persist_disk {
            if let Err(e) = std::fs::create_dir_all(&self.disk_dir) {
                tracing::warn!("Failed to create media disk_dir {:?}: {}", self.disk_dir, e);
            } else {
                if let Err(e) = std::fs::write(self.disk_dir.join(&id), &data) {
                    tracing::warn!("Failed to write media file {}: {}", id, e);
                }
                if let Err(e) = std::fs::write(self.disk_dir.join(format!("{}.mime", id)), mime_type) {
                    tracing::warn!("Failed to write media mime file {}: {}", id, e);
                }
            }
        }

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
        let mode_disk = matches!(&*self.retention.read().unwrap(), RetentionMode::Disk);

        {
            let lock = self.entries.read().await;
            if let Some(entry) = lock.get(id) {
                let ttl = match &*self.retention.read().unwrap() {
                    RetentionMode::Memory(ttl) => Some(*ttl),
                    RetentionMode::Disk => None,
                };
                match ttl {
                    Some(ttl) if entry.created_at.elapsed() >= ttl => return None,
                    _ => return Some((entry.data.clone(), entry.mime_type.clone())),
                }
            }
        }

        if mode_disk {
            let data = std::fs::read(self.disk_dir.join(id)).ok()?;
            let mime = std::fs::read_to_string(self.disk_dir.join(format!("{}.mime", id))).ok()?;
            return Some((data, mime));
        }

        None
    }

    /// Remove expired entries. Call this periodically. No-op in Disk mode.
    pub async fn cleanup_expired(&self) {
        let ttl = match &*self.retention.read().unwrap() {
            RetentionMode::Memory(ttl) => *ttl,
            RetentionMode::Disk => return,
        };
        let mut lock = self.entries.write().await;
        lock.retain(|_, entry| entry.created_at.elapsed() < ttl);
    }
}
