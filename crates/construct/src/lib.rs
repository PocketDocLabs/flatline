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
pub mod auth;
pub mod checkpoint;
pub mod compaction;
pub mod compaction_trigger;
pub mod config;
pub mod context;
pub mod control;
pub mod cost;
pub mod jobs;
pub mod lsp;
pub mod mcp;
pub mod message;
pub mod model_catalog;
pub mod monitors;
pub mod permissions;
pub mod prompt;
pub mod runner;
pub mod s1;
pub mod s2;
pub mod s3;
pub mod s4;
pub mod session;
pub mod shell;
pub mod shells;
pub mod snapshot;
pub mod storage;
pub mod text;
pub mod tool;
pub mod tool_preview;
pub mod topic;
pub mod transcript;
pub mod wakes;
pub mod web;
