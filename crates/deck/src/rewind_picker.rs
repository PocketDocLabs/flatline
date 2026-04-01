#![allow(non_snake_case)]

//! Rewind picker — interactive popup for rewinding the conversation.
//!
//! Shows the current timeline's exchange blocks as rewind targets.
//! User can choose to rewind destructively (Enter) or fork-and-rewind (f)
//! to bookmark the current branch before moving the head.
//!
//! # Public API
//! - [`RewindPicker`] — picker state and rendering
//! - [`RewindAction`] — result of handling a key event
//!
//! # Dependencies
//! `construct::transcript`, `ratatui`

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph, Widget},
};

use construct::transcript::{Turn, TurnRole};

/// Result of handling a key event in the picker.
pub enum RewindAction {
    /// Key consumed, no state change.
    None,
    /// Close the picker.
    Close,
    /// Rewind to a turn ID (destructive — no fork saved).
    Rewind {
        target: String,
        userMessage: String,
        attachments: Option<Vec<construct::transcript::TurnAttachment>>,
    },
    /// Fork current branch, then rewind to a turn ID.
    ForkAndRewind {
        target: String,
        userMessage: String,
        attachments: Option<Vec<construct::transcript::TurnAttachment>>,
    },
}

/// A selectable exchange block in the picker.
struct PickerItem {
    blockId: String,
    userPreview: String,
    assistantPreview: String,
    turnCount: usize,
    /// Turn to rewind to (previous block's last turn).
    rewindTo: String,
    /// Full user message to put back in the input box.
    userMessage: String,
    /// Image attachments from the user turn (for restoring on rewind).
    attachments: Option<Vec<construct::transcript::TurnAttachment>>,
}

/// Interactive rewind picker.
pub struct RewindPicker {
    items: Vec<PickerItem>,
    selected: usize,
    scrollOffset: usize,
}

impl RewindPicker {
    /// Build the picker from branch turns.
    pub fn new(turns: &[Turn]) -> Self {
        let blocks = buildBlocks(turns);

        // Skip blocks with no rewindTo (first block — can't rewind before it).
        let items: Vec<PickerItem> = blocks
            .into_iter()
            .filter(|b| !b.rewindTo.is_empty())
            .map(|b| PickerItem {
                blockId: b.blockId,
                userPreview: b.userPreview,
                assistantPreview: b.assistantPreview,
                turnCount: b.turnCount,
                rewindTo: b.rewindTo,
                userMessage: b.userMessage,
                attachments: b.attachments,
            })
            .collect();

        // Default to second-to-last (most recent rewindable exchange).
        let selected = if items.len() > 1 {
            items.len() - 1
        } else {
            0
        };

        Self {
            items,
            selected,
            scrollOffset: 0,
        }
    }

    /// Handle a key event.
    pub fn handleKey(&mut self, key: KeyEvent) -> RewindAction {
        match key.code {
            KeyCode::Esc => RewindAction::Close,
            KeyCode::Enter => {
                if let Some(item) = self.items.get(self.selected) {
                    RewindAction::Rewind {
                        target: item.rewindTo.clone(),
                        userMessage: item.userMessage.clone(),
                        attachments: item.attachments.clone(),
                    }
                } else {
                    RewindAction::None
                }
            }
            // Fork current branch, then rewind.
            KeyCode::Char('f') => {
                if let Some(item) = self.items.get(self.selected) {
                    RewindAction::ForkAndRewind {
                        target: item.rewindTo.clone(),
                        userMessage: item.userMessage.clone(),
                        attachments: item.attachments.clone(),
                    }
                } else {
                    RewindAction::None
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                RewindAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected + 1 < self.items.len() {
                    self.selected += 1;
                }
                RewindAction::None
            }
            _ => RewindAction::None,
        }
    }

    /// Render the picker as a centered popup overlay.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let popupWidth = (area.width * 7 / 10).max(40).min(area.width.saturating_sub(4));
        let popupHeight = (area.height * 7 / 10).max(10).min(area.height.saturating_sub(2));
        let popupX = area.x + (area.width.saturating_sub(popupWidth)) / 2;
        let popupY = area.y + (area.height.saturating_sub(popupHeight)) / 2;

        let popupArea = Rect {
            x: popupX,
            y: popupY,
            width: popupWidth,
            height: popupHeight,
        };

        let bgStyle = Style::default().bg(Color::Rgb(15, 15, 25)).fg(Color::White);
        for row in popupArea.y..popupArea.y + popupArea.height {
            for col in popupArea.x..popupArea.x + popupArea.width {
                if let Some(cell) = buf.cell_mut((col, row)) {
                    cell.set_char(' ');
                    cell.set_style(bgStyle);
                }
            }
        }

        let borderStyle = Style::default().fg(Color::Cyan).bg(Color::Rgb(15, 15, 25));
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(borderStyle)
            .title(" Rewind ");
        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        if inner.height < 3 || inner.width < 10 {
            return;
        }

        let contentWidth = inner.width as usize;
        let y = inner.y;

        // Footer takes 3 lines (warning + controls + blank).
        let footerHeight: u16 = 3;
        let contentHeight = inner.height.saturating_sub(footerHeight);

        if self.items.is_empty() {
            let emptyStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25));
            renderLine(buf, inner.x + 1, y, contentWidth - 1, "No exchanges to rewind to", emptyStyle);
        } else {
            // Each item is 3 lines tall. Measure layout.
            let itemHeight: u16 = 3;
            let totalHeight = self.items.len() as u16 * itemHeight;

            // Scroll so selected item is visible.
            let selStart = self.selected as u16 * itemHeight;
            let selEnd = selStart + itemHeight;
            if selStart < self.scrollOffset as u16 {
                self.scrollOffset = selStart as usize;
            }
            if selEnd > self.scrollOffset as u16 + contentHeight {
                self.scrollOffset = selEnd.saturating_sub(contentHeight) as usize;
            }

            let scrollOff = self.scrollOffset as u16;

            for (i, item) in self.items.iter().enumerate() {
                let isSelected = i == self.selected;
                let itemY = i as u16 * itemHeight;

                // Skip if entirely off-screen.
                if itemY + itemHeight <= scrollOff || itemY >= scrollOff + contentHeight {
                    continue;
                }

                let blockStyle = if isSelected {
                    Style::default().fg(Color::Cyan).bg(Color::Rgb(40, 40, 80))
                } else {
                    Style::default().fg(Color::Cyan).bg(Color::Rgb(15, 15, 25))
                };
                let userStyle = if isSelected {
                    Style::default().fg(Color::White).bg(Color::Rgb(40, 40, 80))
                } else {
                    Style::default().fg(Color::White).bg(Color::Rgb(15, 15, 25))
                };
                let asstStyle = if isSelected {
                    Style::default().fg(Color::DarkGray).bg(Color::Rgb(40, 40, 80))
                } else {
                    Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25))
                };

                // Clear background for selected item.
                if isSelected {
                    for line in 0..itemHeight {
                        let ry = itemY + line;
                        if ry >= scrollOff && ry < scrollOff + contentHeight {
                            let drawY = y + ry - scrollOff;
                            for col in inner.x..inner.x + inner.width {
                                if let Some(cell) = buf.cell_mut((col, drawY)) {
                                    cell.set_char(' ');
                                    cell.set_style(userStyle);
                                }
                            }
                        }
                    }
                }

                // Line 1: marker + turn count.
                if itemY >= scrollOff && itemY < scrollOff + contentHeight {
                    let drawY = y + itemY - scrollOff;
                    let marker = if isSelected { "\u{25B8}" } else { " " };
                    let text = format!("{marker} Exchange ({} turns)", item.turnCount);
                    renderLine(buf, inner.x, drawY, contentWidth, &truncate(&text, contentWidth), blockStyle);
                }

                // Line 2: user preview.
                let line2Y = itemY + 1;
                if line2Y >= scrollOff && line2Y < scrollOff + contentHeight {
                    let drawY = y + line2Y - scrollOff;
                    let text = format!("  \u{25B9} {}", item.userPreview);
                    renderLine(buf, inner.x, drawY, contentWidth, &truncate(&text, contentWidth), userStyle);
                }

                // Line 3: assistant preview.
                let line3Y = itemY + 2;
                if line3Y >= scrollOff && line3Y < scrollOff + contentHeight {
                    let drawY = y + line3Y - scrollOff;
                    let text = format!("    {}", item.assistantPreview);
                    renderLine(buf, inner.x, drawY, contentWidth, &truncate(&text, contentWidth), asstStyle);
                }
            }

            // Scroll indicator.
            if totalHeight > contentHeight {
                let scrollPct = self.scrollOffset as f64 / (totalHeight - contentHeight).max(1) as f64;
                let trackHeight = contentHeight.saturating_sub(2);
                let thumbY = y + 1 + (scrollPct * trackHeight as f64) as u16;
                let scrollStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25));
                if let Some(cell) = buf.cell_mut((inner.x + inner.width - 1, thumbY)) {
                    cell.set_char('\u{2502}');
                    cell.set_style(scrollStyle);
                }
            }
        }

        // Footer.
        let footerY = popupArea.y + popupArea.height - 3;
        let warningStyle = Style::default().fg(Color::Yellow).bg(Color::Rgb(15, 15, 25));
        let footerStyle = Style::default().fg(Color::DarkGray).bg(Color::Rgb(15, 15, 25));

        renderLine(buf, inner.x, footerY, contentWidth,
            "Enter discards later turns \u{2014} f saves them as a fork first",
            warningStyle);
        renderLine(buf, inner.x, footerY + 1, contentWidth,
            "\u{2191}\u{2193}/jk: navigate  Enter: rewind  f: fork & rewind  Esc: cancel",
            footerStyle);
    }
}

// ---- data helpers ----

struct BlockData {
    blockId: String,
    userPreview: String,
    assistantPreview: String,
    turnCount: usize,
    /// Turn to rewind to (previous block's last turn). Empty for first block.
    rewindTo: String,
    /// Full user message content.
    userMessage: String,
    /// Image attachments from the user turn.
    attachments: Option<Vec<construct::transcript::TurnAttachment>>,
}

fn buildBlocks(turns: &[Turn]) -> Vec<BlockData> {
    if turns.is_empty() {
        return Vec::new();
    }

    let mut blocks: Vec<BlockData> = Vec::new();
    let mut currentBlockId = String::new();
    let mut userPreview = String::new();
    let mut userMessage = String::new();
    let mut userAttachments: Option<Vec<construct::transcript::TurnAttachment>> = None;
    let mut assistantPreview = String::new();
    let mut turnCount: usize = 0;
    let mut lastTurnId = String::new();
    let mut prevBlockLastTurn = String::new();

    for turn in turns {
        if turn.blockId != currentBlockId {
            if !currentBlockId.is_empty() && turnCount > 0 {
                let rewindTo = prevBlockLastTurn.clone();
                blocks.push(BlockData {
                    blockId: currentBlockId.clone(),
                    userPreview: if userPreview.is_empty() { "(no message)".into() } else { userPreview.clone() },
                    assistantPreview: if assistantPreview.is_empty() { "(no response)".into() } else { assistantPreview.clone() },
                    turnCount,
                    rewindTo,
                    userMessage: userMessage.clone(),
                    attachments: userAttachments.clone(),
                });
                prevBlockLastTurn = lastTurnId.clone();
            }
            currentBlockId = turn.blockId.clone();
            userPreview = String::new();
            userMessage = String::new();
            userAttachments = None;
            assistantPreview = String::new();
            turnCount = 0;
        }

        turnCount += 1;
        lastTurnId = turn.id.clone();

        match turn.role {
            TurnRole::User => {
                if userPreview.is_empty() {
                    userPreview = firstLine(&turn.content, 120);
                    userMessage = turn.content.clone();
                    userAttachments = turn.attachments.clone();
                }
            }
            TurnRole::Assistant if assistantPreview.is_empty() => {
                assistantPreview = firstLine(&turn.content, 120);
            }
            _ => {}
        }
    }

    if !currentBlockId.is_empty() && turnCount > 0 {
        blocks.push(BlockData {
            blockId: currentBlockId,
            userPreview: if userPreview.is_empty() { "(no message)".into() } else { userPreview },
            assistantPreview: if assistantPreview.is_empty() { "(no response)".into() } else { assistantPreview },
            turnCount,
            rewindTo: prevBlockLastTurn,
            userMessage,
            attachments: userAttachments,
        });
    }

    blocks
}

fn firstLine(content: &str, maxLen: usize) -> String {
    let line = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    truncate(line, maxLen)
}

fn truncate(s: &str, maxChars: usize) -> String {
    if s.chars().count() <= maxChars {
        s.to_string()
    } else if maxChars > 3 {
        let end = s.char_indices().nth(maxChars - 1).map_or(s.len(), |(i, _)| i);
        format!("{}\u{2026}", &s[..end])
    } else {
        let end = s.char_indices().nth(maxChars).map_or(s.len(), |(i, _)| i);
        s[..end].to_string()
    }
}

fn renderLine(buf: &mut Buffer, x: u16, y: u16, maxWidth: usize, text: &str, style: Style) {
    let area = Rect {
        x,
        y,
        width: maxWidth as u16,
        height: 1,
    };
    Paragraph::new(text.to_string()).style(style).render(area, buf);
}
