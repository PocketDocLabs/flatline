//! Integration test for LSP subsystem.
//!
//! Spawns rust-analyzer against a temporary Rust project,
//! sends a file with a type error, and verifies diagnostics arrive.

#![allow(non_snake_case)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use construct::lsp::LspManager;

const RUN_LSP_INTEGRATION: &str = "FLATLINE_RUN_LSP_INTEGRATION";

fn lspIntegrationEnabled() -> bool {
    if std::env::var_os(RUN_LSP_INTEGRATION).is_some() {
        true
    } else {
        eprintln!("skipping LSP integration test; set {RUN_LSP_INTEGRATION}=1 to run it");
        false
    }
}

/// Create a minimal Rust project in a temp dir for testing.
fn createTestProject() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let projectDir = dir.path().to_path_buf();

    // Cargo.toml
    std::fs::write(
        projectDir.join("Cargo.toml"),
        r#"[package]
name = "lsp-test"
version = "0.1.0"
edition = "2021"
"#,
    )
    .expect("write Cargo.toml");

    // src/main.rs with a type error
    std::fs::create_dir_all(projectDir.join("src")).expect("create src");
    std::fs::write(
        projectDir.join("src/main.rs"),
        r#"fn main() {
    let x: u32 = "not a number";
    println!("{}", x);
}
"#,
    )
    .expect("write main.rs");

    (dir, projectDir)
}

#[tokio::test]
async fn rustAnalyzerDiagnostics() {
    if !lspIntegrationEnabled() {
        return;
    }

    // Skip if rust-analyzer isn't available.
    if std::process::Command::new("rust-analyzer")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("rust-analyzer not found, skipping test");
        return;
    }

    let (_tmpDir, projectDir) = createTestProject();
    let mainPath = projectDir.join("src/main.rs");
    let mainPathStr = mainPath.to_str().unwrap();

    let userConfig = HashMap::new();
    let projectConfig = HashMap::new();
    let mut mgr = LspManager::new(&userConfig, &projectConfig);

    // Touch the file — should lazily spawn rust-analyzer.
    let content = std::fs::read_to_string(mainPathStr).unwrap();
    let hint = mgr.touchFile(mainPathStr, &content).await;

    // Should not hint since rust-analyzer is installed.
    assert!(hint.is_none(), "unexpected hint: {hint:?}");

    // Check server started.
    let statuses = mgr.allServerStatuses();
    let activeCount = statuses
        .iter()
        .filter(|s| matches!(s.status, construct::lsp::ServerAvailability::Active))
        .count();
    assert!(activeCount > 0, "no active servers");
    eprintln!("server statuses: {statuses:?}");

    // Wait for diagnostics (rust-analyzer can take a while on first load).
    let (diags, _hint) = mgr
        .getDiagnostics(mainPathStr, &content, Duration::from_secs(30))
        .await;

    eprintln!("diagnostics output:\n{diags}");

    // Should contain something about the type mismatch.
    assert!(
        !diags.is_empty(),
        "expected diagnostics for type error, got empty"
    );
    assert!(
        diags.contains("ERROR") || diags.contains("error"),
        "expected ERROR in diagnostics: {diags}"
    );

    // Cleanup.
    mgr.shutdown().await;
}

#[tokio::test]
async fn missingServerHint() {
    if !lspIntegrationEnabled() {
        return;
    }

    let userConfig = HashMap::new();
    let projectConfig = HashMap::new();
    let mut mgr = LspManager::new(&userConfig, &projectConfig);

    // Touch a Python file — ty is likely not installed.
    let hint = mgr.touchFile("/tmp/fakefile.py", "x: int = 'hello'").await;

    // Should either connect or hint.
    let statuses = mgr.allServerStatuses();
    let notInstalled = statuses
        .iter()
        .filter(|s| matches!(s.status, construct::lsp::ServerAvailability::NotInstalled))
        .count();

    eprintln!("statuses: {statuses:?}");

    // ty should be in the list and not installed.
    assert!(
        notInstalled > 0 || hint.is_some(),
        "expected either a not-installed server or a hint"
    );

    mgr.shutdown().await;
}
