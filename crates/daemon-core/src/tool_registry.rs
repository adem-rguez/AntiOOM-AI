use std::collections::HashMap;
use tokio::sync::RwLock;
use backend_trait::{Modality, ToolSchema};
use crate::registry::BackendRegistry;

#[derive(Debug, Clone)]
pub struct ToolEntry {
    pub schema: ToolSchema,
    pub modality: Modality,
    pub backend_name: String,
}

pub struct ToolRegistry {
    tools: RwLock<HashMap<String, ToolEntry>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Rebuild the tool map by scanning all backends in the registry.
    /// Call this after every model load/unload event.
    pub async fn refresh(&self, backend_registry: &BackendRegistry) {
        let mut new_tools = HashMap::new();
        // Iterate all known modalities
        let modalities = [
            Modality::Text, Modality::Image, Modality::AudioAsr,
            Modality::AudioTts, Modality::Embedding, Modality::Video,
        ];
        for modality in &modalities {
            if let Some(backend_arc) = backend_registry.get_backend(*modality).await {
                let backend = backend_arc.read().await;
                if backend.is_loaded() {
                    if let Some(schema) = backend.as_tool_schema() {
                        new_tools.insert(schema.name.clone(), ToolEntry {
                            schema,
                            modality: *modality,
                            backend_name: backend.name().to_string(),
                        });
                    }
                }
            }
        }
        let mut lock = self.tools.write().await;
        *lock = new_tools;
    }

    /// Get tool schemas for injection into the calling model's context.
    /// Excludes tools of the given modality (so a text model doesn't see itself).
    pub async fn get_available_tools(&self, exclude_modality: Option<Modality>) -> Vec<ToolSchema> {
        let lock = self.tools.read().await;
        lock.values()
            .filter(|entry| {
                if let Some(exclude) = exclude_modality {
                    entry.modality != exclude
                } else {
                    true
                }
            })
            .map(|entry| entry.schema.clone())
            .collect()
    }

    /// Resolve a tool name to its entry (modality + backend info).
    pub async fn resolve(&self, tool_name: &str) -> Option<ToolEntry> {
        let lock = self.tools.read().await;
        lock.get(tool_name).cloned()
    }
}
