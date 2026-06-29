#![allow(non_snake_case)]

//! Construct — headless agent engine.
//!
//! Handles LLM communication, tool execution, and session state.
//! Designed to run standalone (headless) or be driven by a TUI harness.
//!
//! # Public API
//! - [`config::Config`] — user configuration
//! - [`message`] — message and event types
//! - [`session::Session`] — agent session driver
//! - [`tool`] — tool definitions and execution
//!
//! # Dependencies
//! `reqwest`, `tokio`, `serde`, `serde_json`

pub(crate) mod api;
pub mod auth;
pub(crate) mod auto_review;
pub(crate) mod checkpoint;
pub(crate) mod compaction;
pub(crate) mod compaction_trigger;
pub mod config;
pub mod context;
pub mod control;
pub mod cost;
pub mod jobs;
pub mod lsp;
pub mod mcp;
pub mod message;
pub mod model_catalog;
pub(crate) mod monitors;
pub mod permissions;
pub mod prompt;
pub mod runner;
pub(crate) mod s1;
pub(crate) mod s2;
pub(crate) mod s3;
pub(crate) mod s4;
pub mod session;
pub mod shell;
pub mod shells;
pub mod snapshot;
pub mod storage;
pub mod text;
pub mod tool;
pub(crate) mod tool_preview;
pub mod topic;
pub mod transcript;
pub(crate) mod transcript_search;
pub mod wakes;
pub(crate) mod web;

#[cfg(test)]
mod compaction_pipeline_tests;
