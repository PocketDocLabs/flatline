#![allow(non_snake_case)]

//! Construct — headless agent engine.
//!
//! Handles LLM communication, tool execution, and session state.
//! Designed to run standalone (headless) or be driven by a TUI harness.
//!
//! # Public API
//! - [`config::Config`] — user configuration
//! - [`api::Client`] — OpenRouter API client
//! - [`message`] — message and event types
//! - [`tool`] — tool definitions and execution
//!
//! # Dependencies
//! `reqwest`, `tokio`, `serde`, `serde_json`

pub mod api;
pub mod config;
pub mod message;
pub mod permissions;
pub mod session;
pub mod shell;
pub mod tool;
