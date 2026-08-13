use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    Queued,
    Generating,
    Completed,
    Detached,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSession {
    pub session_id: String,
    pub model_id: String,
    pub status: SessionStatus,
    pub prompt: String,
    pub generated_tokens: u32,
    pub start_timestamp: u64,
}

pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<String, ActiveSession>>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn create_session(&self, session_id: String, model_id: String, prompt: String) -> ActiveSession {
        let session = ActiveSession {
            session_id: session_id.clone(),
            model_id,
            status: SessionStatus::Queued,
            prompt,
            generated_tokens: 0,
            start_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let mut lock = self.sessions.lock().await;
        lock.insert(session_id.clone(), session.clone());
        info!("Created active session '{}'", session_id);
        session
    }

    pub async fn update_status(&self, session_id: &str, status: SessionStatus) {
        let mut lock = self.sessions.lock().await;
        if let Some(session) = lock.get_mut(session_id) {
            session.status = status;
        }
    }

    pub async fn list_sessions(&self) -> Vec<ActiveSession> {
        let lock = self.sessions.lock().await;
        lock.values().cloned().collect()
    }
}
