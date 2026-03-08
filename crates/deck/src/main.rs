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
mod markdown;
mod selection;
mod terminal;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // File-based logging so it doesn't collide with the TUI.
    let logFile = std::fs::File::create("flatline.log")?;
    tracing_subscriber::fmt()
        .with_writer(logFile)
        .with_ansi(false)
        .init();

    app::run().await
}
