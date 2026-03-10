#![allow(non_snake_case)]

//! Deck — TUI harness for flatline agents.
//!
//! Boots a ratatui TUI with an embedded terminal emulator
//! and agent interface.
//!
//! # Public API
//! Binary entry point only.
//!
//! # Dependencies
//! `ratatui`, `crossterm`, `tokio`, `alacritty_terminal`

mod agent_panel;
mod app;
mod history;
mod markdown;
mod selection;
mod terminal;
mod text_area;
mod throbber;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // File-based logging so it doesn't collide with the TUI.
    // Override with FLATLINE_LOG env var (e.g. "trace", "construct=trace,deck=debug").
    let logFile = std::fs::File::create("flatline.log")?;
    let envFilter = tracing_subscriber::EnvFilter::try_from_env("FLATLINE_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug"));
    tracing_subscriber::fmt()
        .with_writer(logFile)
        .with_ansi(false)
        .with_env_filter(envFilter)
        .init();

    app::run().await
}
