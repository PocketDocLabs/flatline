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
pub mod checkpoint;
pub mod compaction;
pub mod control;
pub mod cost;
pub mod compaction_trigger;
pub mod config;
pub mod context;
pub mod lsp;
pub mod mcp;
pub mod message;
pub mod permissions;
pub mod prompt;
pub mod runner;
pub mod s1;
pub mod s2;
pub mod s3;
pub mod s4;
pub mod session;
pub mod shell;
pub mod snapshot;
pub mod text;
pub mod tool;
pub mod topic;
pub mod transcript;
pub mod web;
