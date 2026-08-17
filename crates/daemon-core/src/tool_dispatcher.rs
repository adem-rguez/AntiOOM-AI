use std::sync::Arc;
use std::time::Duration;
use backend_trait::{InferenceRequest, Modality, SamplingParams, ToolCall, ToolResult};
use crate::registry::BackendRegistry;
use crate::tool_registry::ToolRegistry;
use crate::media_store::MediaStore;
use tracing::{info, warn};

pub struct ToolDispatcher {
    tool_registry: Arc<ToolRegistry>,
    backend_registry: Arc<BackendRegistry>,
    media_store: Arc<MediaStore>,
    timeout: Duration,
}

impl ToolDispatcher {
    pub fn new(
        tool_registry: Arc<ToolRegistry>,
        backend_registry: Arc<BackendRegistry>,
        media_store: Arc<MediaStore>,
    ) -> Self {
        Self {
            tool_registry,
            backend_registry,
            media_store,
            timeout: Duration::from_secs(120),
        }
    }

    /// Dispatch a tool call to the appropriate backend.
    /// `depth` tracks call nesting — pass 0 for the initial call.
    pub async fn dispatch(&self, tool_call: &ToolCall, depth: u32) -> ToolResult {
        // Anti-loop: reject chained tool calls (v1 = single-hop only)
        if depth > 0 {
            return ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: "Chained tool calls are not supported in v1. Only single-hop tool calls are allowed.".into(),
                media_handle: None,
                media_data: None,
                media_type: None,
                is_error: true,
            };
        }

        // Resolve tool name to backend
        let entry = match self.tool_registry.resolve(&tool_call.name).await {
            Some(e) => e,
            None => {
                return ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Unknown tool: '{}'. Available tools may have been unloaded.", tool_call.name),
                    media_handle: None,
                    media_data: None,
                    media_type: None,
                    is_error: true,
                };
            }
        };

        // Get backend
        let backend_arc = match self.backend_registry.get_backend(entry.modality).await {
            Some(b) => b,
            None => {
                return ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Backend for modality {:?} is not available.", entry.modality),
                    media_handle: None,
                    media_data: None,
                    media_type: None,
                    is_error: true,
                };
            }
        };

        // Build inference request from tool call arguments
        let prompt = tool_call.arguments.get("prompt")
            .or_else(|| tool_call.arguments.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let request_id = format!("tool-{}-{}", tool_call.name, uuid::Uuid::new_v4());
        let inf_req = InferenceRequest {
            request_id,
            prompt,
            messages: None,
            sampling: SamplingParams::default(),
            modality: entry.modality,
            image_input: None,
            tools: None,
            tool_choice: None,
        };

        info!("Dispatching tool call '{}' to backend '{}'", tool_call.name, entry.backend_name);

        // Execute with timeout
        let backend = backend_arc.read().await;
        match tokio::time::timeout(self.timeout, backend.generate(inf_req)).await {
            Ok(Ok(response)) => {
                // If the response has binary data (media), store it
                if let Some(data) = response.output_data {
                    let mime_type = match entry.modality {
                        Modality::Image => "image/png",
                        Modality::AudioTts => "audio/wav",
                        Modality::AudioAsr => "text/plain",
                        Modality::Video => "video/mp4",
                        _ => "application/octet-stream",
                    };
                    let media_id = self.media_store.store(data, mime_type).await;
                    let handle = format!("/v1/media/{}", media_id);
                    info!("Tool '{}' produced media result: {}", tool_call.name, handle);
                    ToolResult {
                        tool_call_id: tool_call.id.clone(),
                        content: response.output_text,
                        media_handle: Some(handle),
                        media_data: None,
                        media_type: Some(mime_type.to_string()),
                        is_error: false,
                    }
                } else {
                    ToolResult {
                        tool_call_id: tool_call.id.clone(),
                        content: response.output_text,
                        media_handle: None,
                        media_data: None,
                        media_type: None,
                        is_error: false,
                    }
                }
            }
            Ok(Err(e)) => {
                warn!("Tool '{}' backend error: {}", tool_call.name, e);
                ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Tool execution failed: {}", e),
                    media_handle: None,
                    media_data: None,
                    media_type: None,
                    is_error: true,
                }
            }
            Err(_) => {
                warn!("Tool '{}' timed out after {:?}", tool_call.name, self.timeout);
                ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Tool '{}' timed out after {} seconds", tool_call.name, self.timeout.as_secs()),
                    media_handle: None,
                    media_data: None,
                    media_type: None,
                    is_error: true,
                }
            }
        }
    }
}
