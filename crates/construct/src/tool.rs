//! Tool definitions and execution.
//!
//! Defines the tools available to the agent and handles execution.
//! Shell commands go through the construct-owned Shell session.
//! File operations run in-process.
//!
//! # Public API
//! - [`builtinDefs`] — returns tool definitions for the LLM
//! - [`execute`] — execute a parsed tool action
//!
//! # Dependencies
//! `serde_json`

use crate::message::ToolDef;
use crate::shell::Shell;

/// Returns the built-in tool definitions to send to the LLM.
pub fn builtinDefs() -> Vec<ToolDef> {
    vec![
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "shell".into(),
                description: "Execute a shell command and return its output.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute."
                        }
                    },
                    "required": ["command"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "readFile".into(),
                description: "Read the contents of a file.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDef {
            defType: "function".into(),
            function: crate::message::FunctionDef {
                name: "writeFile".into(),
                description: "Write content to a file, creating it if needed.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file."
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
    ]
}

/// A parsed tool invocation from the LLM.
#[derive(Debug)]
pub enum ToolAction {
    Shell { command: String },
    ReadFile { path: String },
    WriteFile { path: String, content: String },
    Unknown { name: String, args: String },
}

/// Human-readable summary of what a tool action will do.
pub fn summarize(action: &ToolAction) -> String {
    match action {
        ToolAction::Shell { command } => format!("Run: {command}"),
        ToolAction::ReadFile { path } => format!("Read: {path}"),
        ToolAction::WriteFile { path, content } => {
            format!("Write {} bytes to {path}", content.len())
        }
        ToolAction::Unknown { name, .. } => format!("Unknown tool: {name}"),
    }
}

/// Execute a tool action and return the output string.
pub async fn execute(action: &ToolAction, shell: &Shell) -> String {
    match action {
        ToolAction::Shell { command } => shell.execute(command).await,
        ToolAction::ReadFile { path } => executeReadFile(path),
        ToolAction::WriteFile { path, content } => executeWriteFile(path, content),
        ToolAction::Unknown { name, .. } => format!("Unknown tool: {name}"),
    }
}

fn executeReadFile(path: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => format!("Failed to read file: {e}"),
    }
}

fn executeWriteFile(path: &str, content: &str) -> String {
    // Create parent directories if needed.
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return format!("Failed to create directories: {e}");
            }
        }
    }
    match std::fs::write(path, content) {
        Ok(()) => format!("Wrote {} bytes to {path}", content.len()),
        Err(e) => format!("Failed to write file: {e}"),
    }
}

/// Parse a tool call name + JSON arguments into a ToolAction.
pub fn parse(name: &str, argsJson: &str) -> ToolAction {
    match name {
        "shell" => {
            let args: serde_json::Value =
                serde_json::from_str(argsJson).unwrap_or_default();
            ToolAction::Shell {
                command: args["command"].as_str().unwrap_or("").into(),
            }
        }
        "readFile" => {
            let args: serde_json::Value =
                serde_json::from_str(argsJson).unwrap_or_default();
            ToolAction::ReadFile {
                path: args["path"].as_str().unwrap_or("").into(),
            }
        }
        "writeFile" => {
            let args: serde_json::Value =
                serde_json::from_str(argsJson).unwrap_or_default();
            ToolAction::WriteFile {
                path: args["path"].as_str().unwrap_or("").into(),
                content: args["content"].as_str().unwrap_or("").into(),
            }
        }
        _ => ToolAction::Unknown {
            name: name.into(),
            args: argsJson.into(),
        },
    }
}
