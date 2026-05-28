#![allow(non_snake_case)]

//! Ephemeral toast overlay.
//!
//! Renders top-middle notifications after the base terminal/agent panels have
//! drawn, without changing layout, scroll position, or transcript content.

use std::time::{Duration, Instant};

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::log_panel::{LogLevel, LogRecord};

const BG: Color = Color::Rgb(31, 31, 38);
const FG_PRIMARY: Color = Color::White;
const FG_DIM: Color = Color::Rgb(140, 140, 160);
const FG_BORDER: Color = Color::Rgb(130, 210, 210);
const FG_OK: Color = Color::Green;
const FG_WARN: Color = Color::Yellow;
const FG_ERR: Color = Color::Red;
const FG_DEBUG: Color = Color::Rgb(100, 100, 120);

struct Toast {
    level: LogLevel,
    title: String,
    detail: Option<String>,
    createdAt: Instant,
    ttl: Duration,
    repeatCount: usize,
}

pub struct ToastCenter {
    toasts: Vec<Toast>,
    visibleLimit: usize,
}

impl ToastCenter {
    pub fn new() -> Self {
        Self {
            toasts: Vec::new(),
            visibleLimit: 3,
        }
    }

    pub fn push(&mut self, record: &LogRecord) {
        if record.level == LogLevel::Debug {
            return;
        }

        if let Some(existing) = self
            .toasts
            .iter_mut()
            .find(|toast| toast.title == record.title && toast.detail == record.detail)
        {
            existing.createdAt = Instant::now();
            existing.repeatCount += 1;
            existing.ttl = ttlFor(record.level);
            return;
        }

        self.toasts.push(Toast {
            level: record.level,
            title: record.title.clone(),
            detail: record.detail.clone(),
            createdAt: Instant::now(),
            ttl: ttlFor(record.level),
            repeatCount: 1,
        });
        let maxKept = self.visibleLimit.saturating_mul(2).max(4);
        if self.toasts.len() > maxKept {
            let excess = self.toasts.len() - maxKept;
            self.toasts.drain(0..excess);
        }
    }

    /// Drop expired toasts. Returns true when visible state changed.
    pub fn tick(&mut self) -> bool {
        let before = self.toasts.len();
        let now = Instant::now();
        self.toasts
            .retain(|toast| now.duration_since(toast.createdAt) < toast.ttl);
        before != self.toasts.len()
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if self.toasts.is_empty() || area.width < 20 || area.height < 6 {
            return;
        }

        let maxWidth = area.width.saturating_sub(4).clamp(20, 84);
        let mut y = area.y.saturating_add(2);

        for toast in self.toasts.iter().rev().take(self.visibleLimit) {
            let width = toastWidth(toast, maxWidth as usize);
            let height = if toast.detail.is_some() { 2 } else { 1 };
            if y + height > area.y + area.height {
                break;
            }
            let rect = Rect {
                x: area.x + area.width.saturating_sub(width) / 2,
                y,
                width,
                height,
            };
            self.renderOne(toast, rect, buf);
            y = y.saturating_add(height);
        }
    }

    fn renderOne(&self, toast: &Toast, rect: Rect, buf: &mut Buffer) {
        fillRect(buf, rect, Style::default().bg(BG).fg(FG_PRIMARY));
        if rect.width == 0 || rect.height == 0 {
            return;
        }

        let icon = iconFor(toast.level);
        let repeat = if toast.repeatCount > 1 {
            format!(" ×{}", toast.repeatCount)
        } else {
            String::new()
        };
        let titleMax = rect.width.saturating_sub(5) as usize;
        let title = truncate(&format!("{}{}", toast.title, repeat), titleMax);
        let line = Line::from(vec![
            Span::styled(
                format!(" {icon} "),
                Style::default().fg(colorFor(toast.level)).bg(BG),
            ),
            Span::styled(title, Style::default().fg(FG_PRIMARY).bg(BG)),
        ]);
        Paragraph::new(line).render(
            Rect {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: 1,
            },
            buf,
        );

        if let Some(detail) = toast.detail.as_deref() {
            let detail = truncate(detail, rect.width.saturating_sub(4) as usize);
            Paragraph::new(Line::from(Span::styled(
                format!("   {detail}"),
                Style::default().fg(FG_DIM).bg(BG),
            )))
            .render(
                Rect {
                    x: rect.x,
                    y: rect.y + 1,
                    width: rect.width,
                    height: 1,
                },
                buf,
            );
        }
    }
}

fn ttlFor(level: LogLevel) -> Duration {
    match level {
        LogLevel::Error => Duration::from_secs(8),
        LogLevel::Warning => Duration::from_secs(7),
        LogLevel::Success => Duration::from_secs(5),
        LogLevel::Info | LogLevel::Debug => Duration::from_secs(4),
    }
}

fn iconFor(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Debug => "·",
        LogLevel::Info => "i",
        LogLevel::Success => "✓",
        LogLevel::Warning => "!",
        LogLevel::Error => "×",
    }
}

fn colorFor(level: LogLevel) -> Color {
    match level {
        LogLevel::Debug => FG_DEBUG,
        LogLevel::Info => FG_BORDER,
        LogLevel::Success => FG_OK,
        LogLevel::Warning => FG_WARN,
        LogLevel::Error => FG_ERR,
    }
}

fn toastWidth(toast: &Toast, max: usize) -> u16 {
    let repeatLen = if toast.repeatCount > 1 {
        format!(" ×{}", toast.repeatCount).chars().count()
    } else {
        0
    };
    let titleLen = 4 + toast.title.chars().count() + repeatLen;
    let detailLen = toast
        .detail
        .as_deref()
        .map(|detail| 4 + detail.chars().count())
        .unwrap_or(0);
    titleLen.max(detailLen).min(max).max(20) as u16
}

fn fillRect(buf: &mut Buffer, area: Rect, style: Style) {
    for row in area.y..area.y + area.height {
        for col in area.x..area.x + area.width {
            if let Some(cell) = buf.cell_mut((col, row)) {
                cell.set_char(' ');
                cell.set_style(style);
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    if max == 1 {
        return "…".to_string();
    }
    let mut out: String = s.chars().take(max - 1).collect();
    out.push('…');
    out
}
