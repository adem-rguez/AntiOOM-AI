use backend_trait::{ToolCall, ToolSchema};
use uuid::Uuid;

/// Build a system prompt suffix that instructs the model to use tools via JSON blocks.
pub fn build_tool_prompt(tools: &[ToolSchema]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut prompt = String::from("\n\nYou have access to the following tools:\n\n");
    for tool in tools {
        prompt.push_str(&format!(
            "### {}\n{}\nParameters: {}\n\n",
            tool.name,
            tool.description,
            serde_json::to_string_pretty(&tool.parameters).unwrap_or_default()
        ));
    }
    prompt.push_str(
        "To use a tool, output a fenced JSON block with the tag `tool_call`:\n\
         ```tool_call\n\
         {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n\
         ```\n\
         Wait for the tool result before continuing your response. \
         Only call one tool at a time. Do not chain multiple tool calls.\n"
    );
    prompt
}

/// Parse the model's output text for tool_call fenced blocks.
/// Returns the first valid tool call found, plus the text before it and after it.
pub fn parse_tool_calls(output: &str) -> Option<ParsedToolCall> {
    // Look for ```tool_call ... ``` blocks
    let marker_start = "```tool_call";
    let marker_end = "```";

    let start_idx = output.find(marker_start)?;
    let json_start = start_idx + marker_start.len();
    // Skip any whitespace/newline after the marker
    let remaining = &output[json_start..];
    let end_idx = remaining.find(marker_end)?;
    let json_str = remaining[..end_idx].trim();

    // Parse the JSON
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;

    let name = parsed.get("name")?.as_str()?.to_string();
    let arguments = parsed.get("arguments").cloned().unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    let tool_call = ToolCall {
        id: format!("fallback-{}", Uuid::new_v4()),
        name,
        arguments,
    };

    Some(ParsedToolCall {
        text_before: output[..start_idx].to_string(),
        tool_call,
        text_after: remaining[end_idx + marker_end.len()..].to_string(),
    })
}

pub struct ParsedToolCall {
    pub text_before: String,
    pub tool_call: ToolCall,
    pub text_after: String,
}
