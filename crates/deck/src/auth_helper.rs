use std::io::{self, IsTerminal};
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    Frame, TerminalOptions, Viewport,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Widget},
};
use tokio::task::JoinHandle;

const INLINE_HEIGHT: u16 = 14;
const PANEL_WIDTH: u16 = 74;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Provider {
    OpenAiCodex,
    Future,
}

impl Provider {
    fn all() -> &'static [Provider] {
        &[Provider::OpenAiCodex, Provider::Future]
    }

    fn label(self) -> &'static str {
        match self {
            Provider::OpenAiCodex => "OpenAI Codex",
            Provider::Future => "More OAuth providers",
        }
    }

    fn status(self) -> String {
        match self {
            Provider::OpenAiCodex => openAiCodexStatusLine(),
            Provider::Future => "not configured yet".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Providers,
    OpenAiCodex,
    WaitingForBrowser,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Provider(Provider),
    SignIn,
    Logout,
    Back,
    Quit,
}

pub async fn run() -> Result<()> {
    if !io::stdout().is_terminal() {
        printPlainStatus();
        return Ok(());
    }

    let mut terminal = initTerminal()?;
    let result = AuthMini::new().run(&mut terminal).await;
    ratatui::try_restore()?;
    result?;
    println!("Flatline auth closed.");
    Ok(())
}

fn initTerminal() -> io::Result<ratatui::DefaultTerminal> {
    match ratatui::try_init_with_options(TerminalOptions {
        viewport: Viewport::Inline(INLINE_HEIGHT),
    }) {
        Ok(terminal) => Ok(terminal),
        Err(inlineError) => {
            let _ = ratatui::try_restore();
            let (width, terminalHeight) =
                crossterm::terminal::size().unwrap_or((PANEL_WIDTH, INLINE_HEIGHT));
            let area = Rect::new(0, 0, width.max(1), INLINE_HEIGHT.min(terminalHeight).max(1));
            ratatui::try_init_with_options(TerminalOptions {
                viewport: Viewport::Fixed(area),
            })
            .map_err(|fallbackError| {
                io::Error::new(
                    fallbackError.kind(),
                    format!(
                        "inline auth panel failed to initialize ({inlineError}); fixed panel fallback failed ({fallbackError})"
                    ),
                )
            })
        }
    }
}

struct AuthMini {
    screen: Screen,
    selected: usize,
    notice: Option<String>,
    pendingLogin: Option<PendingLogin>,
}

struct PendingLogin {
    verificationUrl: String,
    userCode: String,
    task: JoinHandle<Result<construct::auth::OpenAiCodexAuth>>,
}

impl AuthMini {
    fn new() -> Self {
        Self {
            screen: Screen::Providers,
            selected: 0,
            notice: None,
            pendingLogin: None,
        }
    }

    async fn run(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        loop {
            self.checkLoginTask().await;
            terminal.draw(|frame| self.render(frame))?;

            if event::poll(Duration::from_millis(80))?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && self.handleKey(key.code).await?
            {
                break;
            }
        }

        if let Some(pending) = self.pendingLogin.take() {
            pending.task.abort();
        }
        terminal.clear()?;
        Ok(())
    }

    async fn handleKey(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Esc | KeyCode::Char('q') => {
                if self.screen == Screen::Providers {
                    return Ok(true);
                }
                if self.screen == Screen::WaitingForBrowser {
                    if let Some(pending) = self.pendingLogin.take() {
                        pending.task.abort();
                    }
                    self.notice = Some("Sign-in cancelled.".to_string());
                }
                self.screen = Screen::Providers;
                self.selected = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => self.moveSelection(-1),
            KeyCode::Down | KeyCode::Char('j') => self.moveSelection(1),
            KeyCode::Enter | KeyCode::Char(' ') => {
                if let Some(action) = self.actions().get(self.selected).copied() {
                    return self.handleAction(action).await;
                }
            }
            _ => {}
        }
        Ok(false)
    }

    async fn handleAction(&mut self, action: Action) -> Result<bool> {
        match action {
            Action::Provider(Provider::OpenAiCodex) => {
                self.screen = Screen::OpenAiCodex;
                self.selected = 0;
                self.notice = None;
            }
            Action::Provider(Provider::Future) => {
                self.notice =
                    Some("Future OAuth providers will appear here when supported.".to_string());
            }
            Action::SignIn => self.startOpenAiCodexLogin().await?,
            Action::Logout => {
                construct::auth::clearOpenAiCodexAuth()?;
                self.notice = Some("Removed OpenAI Codex credentials.".to_string());
            }
            Action::Back => {
                self.screen = Screen::Providers;
                self.selected = 0;
                self.notice = None;
            }
            Action::Quit => return Ok(true),
        }
        Ok(false)
    }

    fn actions(&self) -> Vec<Action> {
        match self.screen {
            Screen::Providers => Provider::all()
                .iter()
                .copied()
                .map(Action::Provider)
                .chain(std::iter::once(Action::Quit))
                .collect(),
            Screen::OpenAiCodex => {
                let mut actions = vec![Action::SignIn];
                if construct::auth::openAiCodexStatus().configured {
                    actions.insert(1, Action::Logout);
                }
                actions.push(Action::Back);
                actions
            }
            Screen::WaitingForBrowser => Vec::new(),
        }
    }

    fn moveSelection(&mut self, delta: isize) {
        let len = self.actions().len();
        if len == 0 {
            return;
        }
        self.selected = (self.selected as isize + delta).rem_euclid(len as isize) as usize;
    }

    async fn startOpenAiCodexLogin(&mut self) -> Result<()> {
        self.screen = Screen::WaitingForBrowser;
        self.notice = Some("Requesting OpenAI device code...".to_string());
        let device = match construct::auth::requestOpenAiCodexDeviceCode().await {
            Ok(device) => device,
            Err(e) => {
                self.screen = Screen::OpenAiCodex;
                self.notice = Some(format!("Sign-in failed: {e}"));
                return Ok(());
            }
        };

        let verificationUrl = device.verificationUrl.clone();
        let userCode = device.userCode.clone();
        let task = tokio::spawn(construct::auth::completeOpenAiCodexDeviceLogin(device));
        self.pendingLogin = Some(PendingLogin {
            verificationUrl,
            userCode,
            task,
        });
        self.notice = None;
        Ok(())
    }

    async fn checkLoginTask(&mut self) {
        let Some(done) = self
            .pendingLogin
            .as_ref()
            .map(|pending| pending.task.is_finished())
        else {
            return;
        };
        if !done {
            return;
        }

        let Some(pending) = self.pendingLogin.take() else {
            return;
        };
        match pending.task.await {
            Ok(Ok(auth)) => {
                let who = auth
                    .email
                    .as_deref()
                    .or(auth.accountId.as_deref())
                    .unwrap_or("OpenAI account");
                let suffix = auth
                    .planType
                    .as_deref()
                    .map(|plan| format!(" ({plan})"))
                    .unwrap_or_default();
                self.notice = Some(format!("Signed in as {who}{suffix}."));
            }
            Ok(Err(e)) => {
                self.notice = Some(format!("Sign-in failed: {e}"));
            }
            Err(e) => {
                self.notice = Some(format!("Sign-in task ended: {e}"));
            }
        }
        self.screen = Screen::OpenAiCodex;
        self.selected = 0;
    }

    fn render(&self, frame: &mut Frame<'_>) {
        let panel = centeredPanel(frame.area());
        let block = Block::default()
            .title(" Flatline Auth ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = block.inner(panel);
        block.render(panel, frame.buffer_mut());

        match self.screen {
            Screen::Providers => self.renderProviders(frame, inner),
            Screen::OpenAiCodex => self.renderOpenAiCodex(frame, inner),
            Screen::WaitingForBrowser => self.renderWaiting(frame, inner),
        }
    }

    fn renderProviders(&self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = vertical(
            area,
            &[
                Constraint::Length(2),
                Constraint::Min(3),
                Constraint::Length(2),
            ],
        );
        Paragraph::new(Line::from(vec![
            Span::styled(
                "OAuth setup",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  Choose a provider to sign in."),
        ]))
        .render(chunks[0], frame.buffer_mut());

        let items: Vec<ListItem<'_>> = Provider::all()
            .iter()
            .copied()
            .map(|provider| {
                ListItem::new(Line::from(vec![
                    Span::raw(format!("{:<22}", provider.label())),
                    Span::styled(provider.status(), Style::default().fg(Color::Gray)),
                ]))
            })
            .chain(std::iter::once(ListItem::new("Quit")))
            .collect();
        renderList(frame, chunks[1], items, self.selected);
        renderHelp(frame, chunks[2], self.notice.as_deref());
    }

    fn renderOpenAiCodex(&self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = vertical(
            area,
            &[
                Constraint::Length(3),
                Constraint::Min(4),
                Constraint::Length(2),
            ],
        );
        Paragraph::new(vec![
            Line::from(Span::styled(
                "OpenAI Codex OAuth",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(format!("Status: {}", openAiCodexStatusLine())),
            Line::from(format!("Store: {}", construct::auth::authPath().display())),
        ])
        .render(chunks[0], frame.buffer_mut());

        let items: Vec<ListItem<'_>> = self
            .actions()
            .iter()
            .copied()
            .map(|action| ListItem::new(actionLabel(action)))
            .collect();
        renderList(frame, chunks[1], items, self.selected);
        renderHelp(frame, chunks[2], self.notice.as_deref());
    }

    fn renderWaiting(&self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = vertical(area, &[Constraint::Length(8), Constraint::Length(2)]);
        let lines = if let Some(pending) = &self.pendingLogin {
            vec![
                Line::from(Span::styled(
                    "OpenAI Codex sign-in",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from("1. Open this URL:"),
                Line::from(Span::styled(
                    format!("   {}", pending.verificationUrl),
                    Style::default().fg(Color::Cyan),
                )),
                Line::from("2. Enter this code:"),
                Line::from(Span::styled(
                    format!("   {}", pending.userCode),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from("3. Finish in the browser. Flatline is waiting here."),
                Line::from(""),
                Line::from("Esc cancels waiting."),
            ]
        } else {
            vec![
                Line::from(Span::styled(
                    "OpenAI Codex sign-in",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(
                    self.notice
                        .as_deref()
                        .unwrap_or("Requesting OpenAI device code..."),
                ),
            ]
        };
        Paragraph::new(lines).render(chunks[0], frame.buffer_mut());
        renderHelp(frame, chunks[1], self.notice.as_deref());
    }
}

fn renderList(frame: &mut Frame<'_>, area: Rect, items: Vec<ListItem<'_>>, selected: usize) {
    let mut state = ListState::default();
    state.select(Some(selected));
    let list = List::new(items).highlight_symbol("> ").highlight_style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_stateful_widget(list, area, &mut state);
}

fn renderHelp(frame: &mut Frame<'_>, area: Rect, notice: Option<&str>) {
    let text = notice.unwrap_or("up/down select  enter open  esc back  q quit");
    Paragraph::new(text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .render(area, frame.buffer_mut());
}

fn centeredPanel(area: Rect) -> Rect {
    let width = PANEL_WIDTH.min(area.width);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y,
        width,
        height: area.height,
    }
}

fn vertical(area: Rect, constraints: &[Constraint]) -> Vec<Rect> {
    Layout::vertical(constraints.to_vec()).split(area).to_vec()
}

fn actionLabel(action: Action) -> &'static str {
    match action {
        Action::Provider(provider) => provider.label(),
        Action::SignIn => "Sign in with browser",
        Action::Logout => "Remove stored credentials",
        Action::Back => "Back",
        Action::Quit => "Quit",
    }
}

fn openAiCodexStatusLine() -> String {
    let status = construct::auth::openAiCodexStatus();
    if !status.configured {
        return "not signed in".to_string();
    }
    let who = status
        .email
        .as_deref()
        .or(status.accountId.as_deref())
        .unwrap_or("signed in");
    let token = if status.expired { "expired" } else { "valid" };
    match status.planType.as_deref() {
        Some(plan) => format!("{who} ({plan}, {token})"),
        None => format!("{who} ({token})"),
    }
}

fn printPlainStatus() {
    println!("openai-codex: {}", openAiCodexStatusLine());
}
