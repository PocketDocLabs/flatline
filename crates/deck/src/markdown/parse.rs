//! Pulldown-cmark event consumer that produces `Vec<MdBlock>`.
//!
//! Processes the pulldown-cmark event stream through a state machine
//! that accumulates inline content, tables, lists, and code blocks
//! into structured `MdBlock` values.
//!
//! # Public API
//! - [`parse`] — parse markdown text into blocks
//!
//! # Dependencies
//! `pulldown-cmark`, `ratatui`

use pulldown_cmark::{CodeBlockKind, Event, Options, Tag, TagEnd};
use ratatui::style::{Modifier, Style};

use super::block::{Alignment, MdBlock, StyledSegment};

/// Parse markdown text into structured blocks.
///
/// Args:
///     text: Markdown source (already cap-unclosed for streaming).
///
/// Returns:
///     Vec<MdBlock>: Ordered sequence of parsed blocks.
pub fn parse(text: &str) -> Vec<MdBlock> {
    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let parser = pulldown_cmark::Parser::new_ext(text, options);
    let mut state = ParseState::new();

    for event in parser {
        state.process(event);
    }

    state.finish()
}

/// Accumulator for table parsing.
struct TableAccum {
    alignments: Vec<Alignment>,
    header: Vec<Vec<StyledSegment>>,
    rows: Vec<Vec<Vec<StyledSegment>>>,
    currentRow: Vec<Vec<StyledSegment>>,
    currentCell: Vec<StyledSegment>,
    inHead: bool,
}

/// Accumulator for list parsing.
struct ListAccum {
    ordered: bool,
    startIndex: Option<u64>,
    items: Vec<Vec<MdBlock>>,
    currentItemBlocks: Vec<MdBlock>,
}

/// Main parser state machine.
struct ParseState {
    blocks: Vec<MdBlock>,
    /// Stack of inline styles (bold, italic, strikethrough).
    styleStack: Vec<Style>,
    /// Current inline segment accumulator.
    segments: Vec<StyledSegment>,
    /// Active code block accumulation.
    codeBlock: Option<(Option<String>, String)>,
    /// Active table accumulation.
    table: Option<TableAccum>,
    /// Stack of active lists (for nesting).
    listStack: Vec<ListAccum>,
    /// Blockquote nesting depth.
    blockquoteDepth: usize,
    /// Accumulated blocks inside blockquotes.
    blockquoteBlocks: Vec<Vec<MdBlock>>,
    /// Current heading level (0 = not in heading).
    headingLevel: u8,
}

impl ParseState {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            styleStack: Vec::new(),
            segments: Vec::new(),
            codeBlock: None,
            table: None,
            listStack: Vec::new(),
            blockquoteDepth: 0,
            blockquoteBlocks: Vec::new(),
            headingLevel: 0,
        }
    }

    /// Get the current merged inline style from the stack.
    fn currentStyle(&self) -> Style {
        let mut style = Style::default();
        for s in &self.styleStack {
            style = style.patch(*s);
        }
        style
    }

    /// Push a text segment with current styling.
    fn pushText(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.segments.push(StyledSegment {
            text: text.to_string(),
            style: self.currentStyle(),
        });
    }

    /// Flush accumulated segments into a block, pushing to the right destination.
    fn flushParagraph(&mut self) {
        if self.segments.is_empty() {
            return;
        }
        let spans = std::mem::take(&mut self.segments);
        let block = if self.headingLevel > 0 {
            MdBlock::Heading {
                level: self.headingLevel,
                spans,
            }
        } else {
            MdBlock::Paragraph { spans }
        };
        self.pushBlock(block);
    }

    /// Push a finished block to the correct destination (list item, blockquote, or top-level).
    fn pushBlock(&mut self, block: MdBlock) {
        if let Some(list) = self.listStack.last_mut() {
            list.currentItemBlocks.push(block);
        } else if self.blockquoteDepth > 0 {
            if let Some(bq) = self.blockquoteBlocks.last_mut() {
                bq.push(block);
            }
        } else {
            self.blocks.push(block);
        }
    }

    fn process(&mut self, event: Event) {
        match event {
            // -- Block starts --
            Event::Start(Tag::Paragraph) => {}

            Event::Start(Tag::Heading { level, .. }) => {
                self.headingLevel = level as u8;
            }

            Event::Start(Tag::CodeBlock(kind)) => {
                let lang = match kind {
                    CodeBlockKind::Fenced(lang) => {
                        let l = lang.trim().to_string();
                        if l.is_empty() { None } else { Some(l) }
                    }
                    CodeBlockKind::Indented => None,
                };
                self.codeBlock = Some((lang, String::new()));
            }

            Event::Start(Tag::Table(aligns)) => {
                self.table = Some(TableAccum {
                    alignments: aligns.into_iter().map(convertAlignment).collect(),
                    header: Vec::new(),
                    rows: Vec::new(),
                    currentRow: Vec::new(),
                    currentCell: Vec::new(),
                    inHead: false,
                });
            }

            Event::Start(Tag::TableHead) => {
                if let Some(t) = &mut self.table {
                    t.inHead = true;
                }
            }

            Event::Start(Tag::TableRow) => {
                if let Some(t) = &mut self.table {
                    t.currentRow = Vec::new();
                }
            }

            Event::Start(Tag::TableCell) => {
                if let Some(t) = &mut self.table {
                    t.currentCell = Vec::new();
                }
            }

            Event::Start(Tag::List(start)) => {
                // Flush any pending paragraph before entering list.
                self.flushParagraph();
                self.listStack.push(ListAccum {
                    ordered: start.is_some(),
                    startIndex: start,
                    items: Vec::new(),
                    currentItemBlocks: Vec::new(),
                });
            }

            Event::Start(Tag::Item) => {
                if let Some(list) = self.listStack.last_mut() {
                    list.currentItemBlocks = Vec::new();
                }
            }

            Event::Start(Tag::BlockQuote(_)) => {
                self.flushParagraph();
                self.blockquoteDepth += 1;
                self.blockquoteBlocks.push(Vec::new());
            }

            Event::Start(Tag::Emphasis) => {
                self.styleStack
                    .push(Style::default().add_modifier(Modifier::ITALIC));
            }

            Event::Start(Tag::Strong) => {
                self.styleStack
                    .push(Style::default().add_modifier(Modifier::BOLD));
            }

            Event::Start(Tag::Strikethrough) => {
                self.styleStack
                    .push(Style::default().add_modifier(Modifier::CROSSED_OUT));
            }

            Event::Start(Tag::Link { dest_url, .. }) => {
                // NOTE: We render link text inline; URL is lost in TUI.
                self.styleStack.push(
                    Style::default()
                        .fg(ratatui::style::Color::Blue)
                        .add_modifier(Modifier::UNDERLINED),
                );
                let _ = dest_url;
            }

            // -- Block ends --
            Event::End(TagEnd::Paragraph) => {
                self.flushParagraph();
            }

            Event::End(TagEnd::Heading(_)) => {
                self.flushParagraph();
                self.headingLevel = 0;
            }

            Event::End(TagEnd::CodeBlock) => {
                if let Some((lang, code)) = self.codeBlock.take() {
                    self.pushBlock(MdBlock::CodeBlock { lang, code });
                }
            }

            Event::End(TagEnd::Table) => {
                if let Some(t) = self.table.take() {
                    self.pushBlock(MdBlock::Table {
                        alignments: t.alignments,
                        header: t.header,
                        rows: t.rows,
                    });
                }
            }

            Event::End(TagEnd::TableHead) => {
                if let Some(t) = &mut self.table {
                    t.header = std::mem::take(&mut t.currentRow);
                    t.inHead = false;
                }
            }

            Event::End(TagEnd::TableRow) => {
                if let Some(t) = &mut self.table {
                    if !t.inHead {
                        let row = std::mem::take(&mut t.currentRow);
                        t.rows.push(row);
                    }
                }
            }

            Event::End(TagEnd::TableCell) => {
                if let Some(t) = &mut self.table {
                    let cell = std::mem::take(&mut t.currentCell);
                    t.currentRow.push(cell);
                }
            }

            Event::End(TagEnd::List(_)) => {
                if let Some(list) = self.listStack.pop() {
                    self.pushBlock(MdBlock::List {
                        ordered: list.ordered,
                        startIndex: list.startIndex,
                        items: list.items,
                    });
                }
            }

            Event::End(TagEnd::Item) => {
                // Flush any trailing paragraph in the item.
                self.flushParagraph();
                if let Some(list) = self.listStack.last_mut() {
                    let itemBlocks = std::mem::take(&mut list.currentItemBlocks);
                    list.items.push(itemBlocks);
                }
            }

            Event::End(TagEnd::BlockQuote(_)) => {
                self.flushParagraph();
                self.blockquoteDepth -= 1;
                if let Some(bqBlocks) = self.blockquoteBlocks.pop() {
                    self.pushBlock(MdBlock::Blockquote { blocks: bqBlocks });
                }
            }

            Event::End(TagEnd::Emphasis)
            | Event::End(TagEnd::Strong)
            | Event::End(TagEnd::Strikethrough)
            | Event::End(TagEnd::Link) => {
                self.styleStack.pop();
            }

            // -- Content events --
            Event::Text(text) => {
                if let Some((_, ref mut code)) = self.codeBlock {
                    code.push_str(&text);
                } else if self.table.is_some() {
                    let style = self.currentStyle();
                    self.table
                        .as_mut()
                        .unwrap()
                        .currentCell
                        .push(StyledSegment {
                            text: text.to_string(),
                            style,
                        });
                } else {
                    self.pushText(&text);
                }
            }

            Event::Code(code) => {
                let style = Style::default().fg(ratatui::style::Color::Yellow);
                if let Some(t) = &mut self.table {
                    t.currentCell.push(StyledSegment {
                        text: code.to_string(),
                        style,
                    });
                } else {
                    self.segments.push(StyledSegment {
                        text: code.to_string(),
                        style,
                    });
                }
            }

            Event::SoftBreak => {
                if self.codeBlock.is_some() {
                    // Ignored in code blocks.
                } else if self.table.is_some() {
                    let style = self.currentStyle();
                    self.table
                        .as_mut()
                        .unwrap()
                        .currentCell
                        .push(StyledSegment {
                            text: " ".to_string(),
                            style,
                        });
                } else {
                    self.pushText(" ");
                }
            }

            Event::HardBreak => {
                self.pushText("\n");
            }

            Event::Rule => {
                self.pushBlock(MdBlock::Rule);
            }

            // Catch-all for events we don't render specially.
            _ => {}
        }
    }

    fn finish(mut self) -> Vec<MdBlock> {
        // Flush any trailing content.
        self.flushParagraph();
        self.blocks
    }
}

/// Convert pulldown-cmark alignment to our Alignment enum.
fn convertAlignment(a: pulldown_cmark::Alignment) -> Alignment {
    match a {
        pulldown_cmark::Alignment::None => Alignment::None,
        pulldown_cmark::Alignment::Left => Alignment::Left,
        pulldown_cmark::Alignment::Center => Alignment::Center,
        pulldown_cmark::Alignment::Right => Alignment::Right,
    }
}
