use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use backend_trait::{InferenceBackend, Modality};

pub struct BackendRegistry {
    backends: RwLock<HashMap<Modality, Arc<RwLock<Box<dyn InferenceBackend>>>>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            backends: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register_backend(
        &self,
        backend: Box<dyn InferenceBackend>,
    ) {
        let modalities = backend.supported_modalities().to_vec();
        let backend_arc = Arc::new(RwLock::new(backend));

        let mut lock = self.backends.write().await;
        for modality in modalities {
            lock.insert(modality, backend_arc.clone());
        }
    }

    pub async fn get_backend(
        &self,
        modality: Modality,
    ) -> Option<Arc<RwLock<Box<dyn InferenceBackend>>>> {
        let lock = self.backends.read().await;
        lock.get(&modality).cloned()
    }

    pub async fn list_backends(&self) -> Vec<String> {
        let lock = self.backends.read().await;
        let mut names = Vec::new();
        for backend in lock.values() {
            let b = backend.read().await;
            if !names.contains(&b.name().to_string()) {
                names.push(b.name().to_string());
            }
        }
        names
    }
}
