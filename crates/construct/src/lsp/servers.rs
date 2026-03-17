//! Static language server definitions.
//!
//! Built-in server configurations for supported languages.
//! Each entry defines the binary, arguments, file extensions,
//! project root markers, and install hints.
//!
//! # Public API
//! - [`ServerDef`] — static server definition
//! - [`BUILTIN_SERVERS`] — all built-in server definitions
//! - [`serverForExtension`] — find servers matching a file extension
//!
//! # Dependencies
//! None (pure data)

/// Static definition for a built-in language server.
#[derive(Debug, Clone)]
pub struct ServerDef {
    /// Unique identifier (e.g. "rust-analyzer", "ty").
    pub id: &'static str,
    /// Binary name or command.
    pub command: &'static str,
    /// Arguments to pass to the binary.
    pub args: &'static [&'static str],
    /// File extensions this server handles (e.g. ".rs", ".py").
    pub extensions: &'static [&'static str],
    /// Files that indicate a project root (e.g. "Cargo.toml").
    pub rootMarkers: &'static [&'static str],
    /// Human-readable install instruction.
    pub installHint: &'static str,
    /// Required runtime, if any (e.g. "node", "jvm", "dotnet").
    pub runtime: Option<&'static str>,
    /// LSP languageId values for each extension (parallel to extensions).
    pub languageIds: &'static [&'static str],
}

/// All built-in language server definitions.
pub const BUILTIN_SERVERS: &[ServerDef] = &[
    // Tier 1 — Rust-native.
    ServerDef {
        id: "rust-analyzer",
        command: "rust-analyzer",
        args: &[],
        extensions: &[".rs"],
        rootMarkers: &["Cargo.toml", "Cargo.lock"],
        installHint: "rustup component add rust-analyzer",
        runtime: None,
        languageIds: &["rust"],
    },
    ServerDef {
        id: "ty",
        command: "ty",
        args: &["server"],
        extensions: &[".py", ".pyi"],
        rootMarkers: &["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
        installHint: "uv tool install ty",
        runtime: None,
        languageIds: &["python", "python"],
    },
    ServerDef {
        id: "biome",
        command: "biome",
        args: &["lsp-proxy"],
        extensions: &[
            ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts",
            ".json", ".jsonc", ".css",
        ],
        rootMarkers: &["biome.json", "biome.jsonc", "package.json"],
        installHint: "npm install -g @biomejs/biome",
        runtime: None,
        languageIds: &[
            "javascript", "javascriptreact", "typescript", "typescriptreact",
            "javascript", "javascript", "typescript", "typescript",
            "json", "jsonc", "css",
        ],
    },
    // Tier 2 — Native binaries.
    ServerDef {
        id: "gopls",
        command: "gopls",
        args: &[],
        extensions: &[".go"],
        rootMarkers: &["go.mod", "go.sum"],
        installHint: "go install golang.org/x/tools/gopls@latest",
        runtime: None,
        languageIds: &["go"],
    },
    ServerDef {
        id: "clangd",
        command: "clangd",
        args: &["--background-index"],
        extensions: &[".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hh", ".hxx"],
        rootMarkers: &["compile_commands.json", "CMakeLists.txt", ".clangd"],
        installHint: "brew install llvm (macOS) / apt install clangd (Linux)",
        runtime: None,
        languageIds: &["c", "c", "cpp", "cpp", "cpp", "cpp", "cpp", "cpp"],
    },
    // Tier 3 — Runtime-dependent.
    ServerDef {
        id: "bash-language-server",
        command: "bash-language-server",
        args: &["start"],
        extensions: &[".sh", ".bash"],
        rootMarkers: &[],
        installHint: "npm install -g bash-language-server",
        runtime: Some("node"),
        languageIds: &["shellscript", "shellscript"],
    },
    ServerDef {
        id: "yaml-language-server",
        command: "yaml-language-server",
        args: &["--stdio"],
        extensions: &[".yaml", ".yml"],
        rootMarkers: &[],
        installHint: "npm install -g yaml-language-server",
        runtime: Some("node"),
        languageIds: &["yaml", "yaml"],
    },
    ServerDef {
        id: "typescript-language-server",
        command: "typescript-language-server",
        args: &["--stdio"],
        extensions: &[".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"],
        rootMarkers: &["tsconfig.json", "jsconfig.json", "package.json"],
        installHint: "npm install -g typescript-language-server typescript",
        runtime: Some("node"),
        languageIds: &[
            "typescript", "typescriptreact", "javascript", "javascriptreact",
            "javascript", "javascript", "typescript", "typescript",
        ],
    },
    ServerDef {
        id: "jdtls",
        command: "jdtls",
        args: &[],
        extensions: &[".java"],
        rootMarkers: &["pom.xml", "build.gradle", "build.gradle.kts", ".project"],
        installHint: "brew install jdtls (macOS) / see https://github.com/eclipse-jdtls/eclipse.jdt.ls",
        runtime: Some("jvm"),
        languageIds: &["java"],
    },
    ServerDef {
        id: "csharp-ls",
        command: "csharp-ls",
        args: &[],
        extensions: &[".cs"],
        rootMarkers: &[".sln", ".csproj"],
        installHint: "dotnet tool install --global csharp-ls",
        runtime: Some("dotnet"),
        languageIds: &["csharp"],
    },
];

/// Find all server definitions whose extensions match the given file extension.
///
/// Args:
///     ext: File extension including the dot (e.g. ".rs", ".py").
///
/// Returns:
///     Vec of matching server definitions.
pub fn serversForExtension(ext: &str) -> Vec<&'static ServerDef> {
    BUILTIN_SERVERS
        .iter()
        .filter(|s| s.extensions.contains(&ext))
        .collect()
}

/// Get the languageId for a file extension from a server definition.
///
/// Args:
///     server: The server definition.
///     ext: File extension including the dot.
///
/// Returns:
///     The LSP languageId string, or "plaintext" if not found.
pub fn languageIdForExtension(server: &ServerDef, ext: &str) -> &'static str {
    server
        .extensions
        .iter()
        .zip(server.languageIds.iter())
        .find(|&(&e, _)| e == ext)
        .map(|(_, &id)| id)
        .unwrap_or("plaintext")
}
