#![allow(non_snake_case)]

//! MCP serve mode — expose flatline's built-in tools as an MCP server.
//!
//! When run with `flatline --serve`, flatline acts as an MCP server
//! over stdio, allowing other MCP clients (Claude Code, Cursor, etc.)
//! to use flatline's tools.
//!
//! Exposes a curated subset of tools: shell, readFile, writeFile,
//! glob, grep, listDir.
//!
//! # Public API
//! - [`run`] — start the MCP server on stdio
//!
//! # Dependencies
//! `rmcp` (server, transport-io), `tokio`

use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
};

use serde::Deserialize;

// -- Parameter structs for each tool --

#[derive(Deserialize, schemars::JsonSchema)]
struct ShellParams {
    /// The shell command to execute.
    command: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ReadFileParams {
    /// Path to the file to read.
    path: String,
    /// Line number to start reading from (1-indexed).
    offset: Option<u64>,
    /// Maximum number of lines to read.
    limit: Option<u64>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct WriteFileParams {
    /// Path to the file to write.
    path: String,
    /// The content to write.
    content: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct GrepParams {
    /// Regular expression pattern to search for.
    pattern: String,
    /// Directory or file to search in. Defaults to current directory.
    path: Option<String>,
    /// Glob pattern to filter files (e.g. "*.rs").
    include: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct GlobParams {
    /// Glob pattern to match (e.g. "**/*.rs", "src/**/*.ts").
    pattern: String,
    /// Directory to search in. Defaults to current directory.
    path: Option<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ListDirParams {
    /// Directory path to list. Defaults to current directory.
    path: Option<String>,
    /// Maximum depth to recurse (1-5). Defaults to 2.
    depth: Option<u64>,
}

/// Flatline MCP server handler.
///
/// Exposes a subset of flatline's built-in tools to MCP clients.
#[derive(Clone)]
pub struct FlatlineServer {
    toolRouter: ToolRouter<Self>,
}

#[tool_router]
impl FlatlineServer {
    fn new() -> Self {
        Self {
            toolRouter: Self::tool_router(),
        }
    }

    #[tool(
        description = "Execute a shell command and return its output. Output is truncated at 2000 lines / 100KB."
    )]
    async fn shell(
        &self,
        Parameters(params): Parameters<ShellParams>,
    ) -> Result<CallToolResult, McpError> {
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&params.command)
            .output()
            .await
            .map_err(|e| {
                McpError::internal_error(format!("Failed to execute command: {e}"), None)
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = stdout.to_string();
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr]\n");
            result.push_str(&stderr);
        }

        // Truncate large output.
        if result.len() > 100_000 {
            result.truncate(100_000);
            result.push_str("\n... [truncated]");
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(description = "Read the contents of a file. Returns numbered lines.")]
    async fn readFile(
        &self,
        Parameters(params): Parameters<ReadFileParams>,
    ) -> Result<CallToolResult, McpError> {
        let content = tokio::fs::read_to_string(&params.path)
            .await
            .map_err(|e| McpError::invalid_params(format!("Failed to read file: {e}"), None))?;

        let offset = params.offset.unwrap_or(1).max(1) as usize;
        let limit = params.limit.unwrap_or(2000) as usize;

        let lines: Vec<&str> = content.lines().collect();
        let start = (offset - 1).min(lines.len());
        let end = (start + limit).min(lines.len());

        let mut output = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            let lineNum = start + i + 1;
            let displayLine = if line.len() > 2000 {
                &line[..line.floor_char_boundary(2000)]
            } else {
                line
            };
            output.push_str(&format!("{lineNum:>6}\t{displayLine}\n"));
        }

        if end < lines.len() {
            output.push_str(&format!(
                "\n... {} more lines. Use offset/limit to read more.\n",
                lines.len() - end
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    #[tool(
        description = "Write content to a file, creating it if needed. Overwrites the entire file."
    )]
    async fn writeFile(
        &self,
        Parameters(params): Parameters<WriteFileParams>,
    ) -> Result<CallToolResult, McpError> {
        // Create parent directories if needed.
        if let Some(parent) = std::path::Path::new(&params.path).parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                McpError::internal_error(format!("Failed to create directory: {e}"), None)
            })?;
        }

        tokio::fs::write(&params.path, &params.content)
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to write file: {e}"), None))?;

        let path = &params.path;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Wrote {} bytes to {path}",
            params.content.len()
        ))]))
    }

    #[tool(
        description = "Search file contents using ripgrep. Returns matching file paths or content lines."
    )]
    async fn grep(
        &self,
        Parameters(params): Parameters<GrepParams>,
    ) -> Result<CallToolResult, McpError> {
        let searchPath = params.path.as_deref().unwrap_or(".");

        let mut cmd = tokio::process::Command::new("rg");
        cmd.arg("--max-count=100")
            .arg("--no-heading")
            .arg("--line-number");

        if let Some(ref glob) = params.include {
            cmd.arg("--glob").arg(glob);
        }

        cmd.arg(&params.pattern).arg(searchPath);

        let output = cmd
            .output()
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to run ripgrep: {e}"), None))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        if stdout.is_empty() {
            let pattern = &params.pattern;
            Ok(CallToolResult::success(vec![Content::text(format!(
                "No matches found for \"{pattern}\" in {searchPath}"
            ))]))
        } else {
            let result = if stdout.len() > 50_000 {
                format!("{}... [truncated]", &stdout[..50_000])
            } else {
                stdout.to_string()
            };
            Ok(CallToolResult::success(vec![Content::text(result)]))
        }
    }

    #[tool(description = "Find files by glob pattern. Returns matching file paths.")]
    async fn glob(
        &self,
        Parameters(params): Parameters<GlobParams>,
    ) -> Result<CallToolResult, McpError> {
        let searchPath = params.path.as_deref().unwrap_or(".");

        let mut cmd = tokio::process::Command::new("rg");
        cmd.arg("--files")
            .arg("--glob")
            .arg(&params.pattern)
            .arg(searchPath);

        let output = cmd
            .output()
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to run ripgrep: {e}"), None))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        if stdout.is_empty() {
            let pattern = &params.pattern;
            Ok(CallToolResult::success(vec![Content::text(format!(
                "No files found matching \"{pattern}\" in {searchPath}"
            ))]))
        } else {
            let lines: Vec<&str> = stdout.lines().take(100).collect();
            let total = stdout.lines().count();
            let mut result = lines.join("\n");
            if total > 100 {
                result.push_str(&format!("\n... and {} more files", total - 100));
            }
            Ok(CallToolResult::success(vec![Content::text(result)]))
        }
    }

    #[tool(description = "List directory contents with tree-like output.")]
    async fn listDir(
        &self,
        Parameters(params): Parameters<ListDirParams>,
    ) -> Result<CallToolResult, McpError> {
        let dirPath = params.path.as_deref().unwrap_or(".");
        let maxDepth = params.depth.unwrap_or(2).min(5).max(1);

        let mut cmd = tokio::process::Command::new("find");
        cmd.arg(dirPath)
            .arg("-maxdepth")
            .arg(maxDepth.to_string())
            .arg("-not")
            .arg("-path")
            .arg("*/.git/*")
            .arg("-not")
            .arg("-path")
            .arg("*/node_modules/*")
            .arg("-not")
            .arg("-path")
            .arg("*/target/*");

        let output = cmd.output().await.map_err(|e| {
            McpError::internal_error(format!("Failed to list directory: {e}"), None)
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().take(200).collect();
        let total = stdout.lines().count();
        let mut result = lines.join("\n");
        if total > 200 {
            result.push_str(&format!("\n... and {} more entries", total - 200));
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}

#[tool_handler(router = self.toolRouter)]
impl ServerHandler for FlatlineServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2025_03_26)
            .with_server_info(Implementation::new("flatline", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Flatline is a general-purpose agent tool. Use the available tools to \
             execute shell commands, read/write files, search code, and explore \
             directory structures.",
            )
    }
}

/// Run the MCP server on stdio.
///
/// Blocks until the client disconnects or an error occurs.
pub async fn run() -> anyhow::Result<()> {
    tracing::info!("starting flatline MCP server on stdio");

    let server = FlatlineServer::new();
    let transport = rmcp::transport::io::stdio();

    let service = server
        .serve(transport)
        .await
        .map_err(|e| anyhow::anyhow!("MCP server initialization failed: {e}"))?;

    tracing::info!("flatline MCP server running");

    // Wait until the service shuts down.
    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;

    tracing::info!("flatline MCP server stopped");
    Ok(())
}
