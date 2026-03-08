//! Embedded terminal emulator module.
//!
//! Uses `alacritty_terminal` for VT state and `ratatui` for rendering.
//! PTY management lives in construct — deck only handles display.
//!
//! # Public API
//! - [`Terminal`] — the ratatui stateful widget
//! - [`TerminalState`] — VT emulator state

pub mod widget;

pub use widget::{Terminal, TerminalState};
