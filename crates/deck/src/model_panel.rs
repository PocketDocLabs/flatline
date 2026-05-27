#![allow(non_snake_case)]

//! Model profile panel: compact UI for viewing and switching configured profiles.

use construct::config::ModelTier;
use construct::control::ModelStatus;
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};

const BG: Color = Color::Rgb(15, 17, 24);
const BG_SELECTED: Color = Color::Rgb(38, 48, 66);
const FG_PRIMARY: Color = Color::White;
const FG_DIM: Color = Color::Rgb(120, 130, 145);
const FG_MUTED: Color = Color::Rgb(72, 80, 96);
const FG_BORDER: Color = Color::Rgb(86, 168, 130);
const FG_GOOD: Color = Color::Rgb(118, 210, 155);
const FG_WARN: Color = Color::Rgb(230, 180, 95);
const HEADER_LINES: u16 = 5;
const PROFILE_HEADER_LINES: u16 = 1;
const FOOTER_RESERVE: u16 = 3;

pub enum PanelAction {
    None,
    Close,
    Save { tier: ModelTier, profile: String },
}

pub struct ModelPanel {
    status: ModelStatus,
    selectedTier: ModelTier,
    selectedProfile: usize,
    scrollOffset: usize,
    lastVisibleCount: usize,
    notice: Option<String>,
}

impl ModelPanel {
    pub fn new(status: ModelStatus) -> Self {
        let selectedProfile = status
            .profiles
            .iter()
            .position(|p| p.name == status.heavyProfile)
            .unwrap_or(0);
        Self {
            status,
            selectedTier: ModelTier::Heavy,
            selectedProfile,
            scrollOffset: 0,
            lastVisibleCount: 5,
            notice: None,
        }
    }

    pub fn handleKey(&mut self, key: KeyEvent) -> PanelAction {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Char('h') | KeyCode::Char('H') => {
                self.setTier(ModelTier::Heavy);
                PanelAction::None
            }
            KeyCode::Char('l') | KeyCode::Char('L') => {
                self.setTier(ModelTier::Light);
                PanelAction::None
            }
            KeyCode::Char('u') | KeyCode::Char('U') => {
                self.setTier(ModelTier::Utility);
                PanelAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selectedProfile > 0 {
                    self.selectedProfile -= 1;
                    self.adjustScroll();
                }
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selectedProfile + 1 < self.status.profiles.len() {
                    self.selectedProfile += 1;
                    self.adjustScroll();
                }
                PanelAction::None
            }
            KeyCode::Enter | KeyCode::Char('s') => {
                let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
                    return PanelAction::None;
                };
                let profile = profile.name.clone();
                self.setActiveProfile(self.selectedTier, profile.clone());
                self.notice = Some(format!(
                    "{} -> {}",
                    tierLabel(self.selectedTier),
                    profile
                ));
                PanelAction::Save {
                    tier: self.selectedTier,
                    profile,
                }
            }
            _ => PanelAction::None,
        }
    }

    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let popupWidth = self.preferredPopupWidth(area);
        let popupHeight = self.preferredPopupHeight(area);
        let popupX = area.x + (area.width.saturating_sub(popupWidth)) / 2;
        let popupY = area.y + (area.height.saturating_sub(popupHeight)) / 2;
        let popupArea = Rect {
            x: popupX,
            y: popupY,
            width: popupWidth,
            height: popupHeight,
        };

        let bgStyle = Style::default().bg(BG).fg(FG_PRIMARY);
        fillRect(buf, popupArea, bgStyle);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FG_BORDER).bg(BG))
            .title(" Model Profiles ");
        let inner = block.inner(popupArea);
        block.render(popupArea, buf);

        if inner.height < 5 || inner.width < 24 {
            return;
        }

        let w = inner.width as usize;
        let mut y = inner.y;
        self.renderHeader(buf, inner.x, w, &mut y);
        self.renderProfiles(buf, inner, w, &mut y);
        self.renderFooter(buf, popupArea, inner, w);
    }

    fn preferredPopupWidth(&self, area: Rect) -> u16 {
        let maxPopup = area.width.saturating_sub(2);
        let maxInner = maxPopup.saturating_sub(2) as usize;
        if maxInner == 0 {
            return maxPopup;
        }

        let minInner = 68usize.min(maxInner);
        let desiredInner = self.preferredInnerWidth().max(minInner).min(maxInner);
        (desiredInner as u16).saturating_add(2).min(maxPopup)
    }

    fn preferredPopupHeight(&self, area: Rect) -> u16 {
        let maxPopup = area.height.saturating_sub(2);
        if maxPopup == 0 {
            return maxPopup;
        }

        let desiredInner = HEADER_LINES
            .saturating_add(PROFILE_HEADER_LINES)
            .saturating_add(self.status.profiles.len() as u16)
            .saturating_add(FOOTER_RESERVE);
        let desiredPopup = desiredInner.saturating_add(2);
        let minPopup = 14u16.min(maxPopup);
        let softMax = ((area.height as u32 * 9 / 10) as u16).min(maxPopup);
        desiredPopup.max(minPopup).min(softMax.max(minPopup))
    }

    fn preferredInnerWidth(&self) -> usize {
        let mut width = " Model Profiles ".chars().count();
        width = width.max(self.tierLine().chars().count());
        width = width.max(self.authLine().chars().count());
        width = width.max(self.selectionLine().chars().count());
        width = width.max(self.saveLine().chars().count());
        width = width.max(self.footerLine().chars().count());
        width.max(self.profileTableNaturalWidth())
    }

    fn setTier(&mut self, tier: ModelTier) {
        self.selectedTier = tier;
        let active = self.activeProfileFor(tier).to_string();
        if let Some(idx) = self.status.profiles.iter().position(|p| p.name == active) {
            self.selectedProfile = idx;
            self.adjustScroll();
        }
    }

    fn activeProfileFor(&self, tier: ModelTier) -> &str {
        match tier {
            ModelTier::Heavy => &self.status.heavyProfile,
            ModelTier::Light => &self.status.lightProfile,
            ModelTier::Utility => &self.status.utilityProfile,
        }
    }

    fn setActiveProfile(&mut self, tier: ModelTier, profile: String) {
        match tier {
            ModelTier::Heavy => self.status.heavyProfile = profile,
            ModelTier::Light => self.status.lightProfile = profile,
            ModelTier::Utility => self.status.utilityProfile = profile,
        }
    }

    fn tierLine(&self) -> String {
        format!(
            " Heavy: {}   Light: {}   Utility: {}",
            self.status.heavyProfile, self.status.lightProfile, self.status.utilityProfile
        )
    }

    fn authLine(&self) -> String {
        let auth = &self.status.openAiCodex;
        if auth.configured {
            let who = auth
                .email
                .as_deref()
                .or(auth.accountId.as_deref())
                .unwrap_or("signed in");
            let plan = auth.planType.as_deref().unwrap_or("plan unknown");
            if auth.expired {
                format!(" OpenAI Codex: {who} ({plan}), token expired")
            } else {
                format!(" OpenAI Codex: {who} ({plan})")
            }
        } else {
            " OpenAI Codex: not signed in".to_string()
        }
    }

    fn selectionLine(&self) -> String {
        if let Some(notice) = &self.notice {
            format!(" Saved: {notice}")
        } else {
            format!(
                " Editing {} tier. Enter saves and leaves this panel open.",
                tierLabel(self.selectedTier)
            )
        }
    }

    fn saveLine(&self) -> String {
        format!(" Saves to {}", self.status.configPath)
    }

    fn footerLine(&self) -> String {
        format!(
            " h/l/u tier: {}   up/down select   enter save   * active profile   q close ",
            tierLabel(self.selectedTier)
        )
    }

    fn profileTableNaturalWidth(&self) -> usize {
        let mut width = profileNaturalWidth("  ", "profile", "provider", "model", "ctx", "state");
        for profile in &self.status.profiles {
            let state = if profile.configured {
                "ready"
            } else {
                "needs auth"
            };
            let ctx = format!("{}k", profile.contextWindow / 1000);
            width = width.max(profileNaturalWidth(
                ">*",
                &profile.name,
                &profile.provider,
                &profile.model,
                &ctx,
                state,
            ));
        }
        width
    }

    fn profileColumns(&self, width: usize) -> ProfileColumns {
        let mut columns = ProfileColumns {
            marker: 2,
            name: "profile".chars().count(),
            provider: "provider".chars().count(),
            model: "model".chars().count(),
            context: "ctx".chars().count().max(7),
            state: "state".chars().count().max("needs auth".chars().count()),
        };

        for profile in &self.status.profiles {
            columns.name = columns.name.max(profile.name.chars().count());
            columns.provider = columns.provider.max(profile.provider.chars().count());
            columns.model = columns.model.max(profile.model.chars().count());
            columns.context = columns
                .context
                .max(format!("{}k", profile.contextWindow / 1000).chars().count());
        }

        let min = ProfileColumns {
            marker: 2,
            name: 12,
            provider: 9,
            model: 8,
            context: 7,
            state: 10,
        };
        columns.fitTo(width, min)
    }

    fn renderHeader(&self, buf: &mut Buffer, x: u16, w: usize, y: &mut u16) {
        let tierLine = self.tierLine();
        line(
            buf,
            x,
            *y,
            w,
            &truncateStr(&tierLine, w),
            style(FG_PRIMARY, BG),
        );
        *y += 1;

        let auth = &self.status.openAiCodex;
        let authLabel = self.authLine();
        line(
            buf,
            x,
            *y,
            w,
            &truncateStr(&authLabel, w),
            style(if auth.configured { FG_GOOD } else { FG_WARN }, BG),
        );
        *y += 1;

        let selectionLine = self.selectionLine();
        let selectionStyle = if self.notice.is_some() {
            style(FG_GOOD, BG)
        } else {
            style(FG_DIM, BG)
        };
        line(
            buf,
            x,
            *y,
            w,
            &truncateStr(&selectionLine, w),
            selectionStyle,
        );
        *y += 1;

        let saveLine = self.saveLine();
        line(buf, x, *y, w, &truncateStr(&saveLine, w), style(FG_DIM, BG));
        *y += 1;

        let sep: String = "-".repeat(w.saturating_sub(2));
        line(buf, x + 1, *y, w - 2, &sep, style(FG_MUTED, BG));
        *y += 1;
    }

    fn renderProfiles(&mut self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        let columns = self.profileColumns(w);
        line(
            buf,
            inner.x,
            *y,
            w,
            &profileRow(columns, "  ", "profile", "provider", "model", "ctx", "state", w),
            style(FG_DIM, BG),
        );
        *y += 1;

        let available = (inner.y + inner.height).saturating_sub(*y + FOOTER_RESERVE) as usize;
        let visibleCount =
            available.min(self.status.profiles.len().saturating_sub(self.scrollOffset));
        self.lastVisibleCount = visibleCount.max(1);

        let selectedTier = self.selectedTier;
        let active = self.activeProfileFor(selectedTier).to_string();
        for i in 0..visibleCount {
            let idx = self.scrollOffset + i;
            let Some(profile) = self.status.profiles.get(idx) else {
                break;
            };
            let selected = idx == self.selectedProfile;
            let bg = if selected { BG_SELECTED } else { BG };
            let marker = if selected { ">" } else { " " };
            let activeMarker = if profile.name == active { "*" } else { " " };
            let configured = if profile.configured {
                "ready"
            } else {
                "needs auth"
            };
            let ctx = format!("{}k", profile.contextWindow / 1000);
            let text = profileRow(
                columns,
                &format!("{marker}{activeMarker}"),
                &profile.name,
                &profile.provider,
                &profile.model,
                &ctx,
                configured,
                w,
            );
            line(
                buf,
                inner.x,
                *y,
                w,
                &truncateStr(&text, w),
                style(FG_PRIMARY, bg),
            );
            *y += 1;
        }

        while *y < inner.y + inner.height - FOOTER_RESERVE {
            line(buf, inner.x, *y, w, "", style(FG_PRIMARY, BG));
            *y += 1;
        }
    }

    fn renderFooter(&self, buf: &mut Buffer, popup: Rect, inner: Rect, w: usize) {
        let y = popup.y + popup.height.saturating_sub(2);
        let footer = self.footerLine();
        line(
            buf,
            inner.x,
            y,
            w,
            &truncateStr(&footer, w),
            style(FG_DIM, BG),
        );
    }

    fn adjustScroll(&mut self) {
        if self.selectedProfile < self.scrollOffset {
            self.scrollOffset = self.selectedProfile;
        } else if self.selectedProfile >= self.scrollOffset + self.lastVisibleCount {
            self.scrollOffset = self
                .selectedProfile
                .saturating_sub(self.lastVisibleCount.saturating_sub(1));
        }
    }
}

fn fillRect(buf: &mut Buffer, area: Rect, style: Style) {
    for y in area.y..area.y.saturating_add(area.height) {
        for x in area.x..area.x.saturating_add(area.width) {
            buf[(x, y)].set_style(style);
            buf[(x, y)].set_symbol(" ");
        }
    }
}

fn line(buf: &mut Buffer, x: u16, y: u16, w: usize, text: &str, style: Style) {
    let mut col = x;
    for ch in text.chars().take(w) {
        buf[(col, y)].set_char(ch).set_style(style);
        col = col.saturating_add(1);
    }
    for _ in text.chars().count()..w {
        buf[(col, y)].set_char(' ').set_style(style);
        col = col.saturating_add(1);
    }
}

fn style(fg: Color, bg: Color) -> Style {
    Style::default().fg(fg).bg(bg)
}

fn tierLabel(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Heavy => "heavy",
        ModelTier::Light => "light",
        ModelTier::Utility => "utility",
    }
}

#[derive(Clone, Copy)]
struct ProfileColumns {
    marker: usize,
    name: usize,
    provider: usize,
    model: usize,
    context: usize,
    state: usize,
}

impl ProfileColumns {
    fn total(self) -> usize {
        self.marker + self.name + self.provider + self.model + self.context + self.state + 5
    }

    fn fitTo(mut self, width: usize, min: ProfileColumns) -> ProfileColumns {
        while self.total() > width && self.model > min.model {
            self.model -= 1;
        }
        while self.total() > width && self.name > min.name {
            self.name -= 1;
        }
        while self.total() > width && self.provider > min.provider {
            self.provider -= 1;
        }
        while self.total() > width && self.context > min.context {
            self.context -= 1;
        }
        while self.total() > width && self.state > min.state {
            self.state -= 1;
        }

        let mut extra = width.saturating_sub(self.total());

        let nameTarget = self.name.max(24);
        let nameExtra = extra.min(nameTarget.saturating_sub(self.name));
        self.name += nameExtra;
        extra -= nameExtra;

        let providerTarget = self.provider.max(16);
        let providerExtra = extra.min(providerTarget.saturating_sub(self.provider));
        self.provider += providerExtra;
        extra -= providerExtra;

        self.model += extra;
        self
    }
}

fn profileNaturalWidth(
    marker: &str,
    name: &str,
    provider: &str,
    model: &str,
    context: &str,
    state: &str,
) -> usize {
    marker.chars().count()
        + name.chars().count()
        + provider.chars().count()
        + model.chars().count()
        + context.chars().count()
        + state.chars().count()
        + 5
}

fn profileRow(
    columns: ProfileColumns,
    marker: &str,
    name: &str,
    provider: &str,
    model: &str,
    context: &str,
    state: &str,
    width: usize,
) -> String {
    if width < columns.total() {
        return truncateStr(
            &format!("{marker} {name}  {provider}  {model}  {state}"),
            width,
        );
    }

    truncateStr(
        &format!(
            "{} {} {} {} {} {}",
            padCell(marker, columns.marker),
            padCell(name, columns.name),
            padCell(provider, columns.provider),
            padCell(model, columns.model),
            padCell(context, columns.context),
            padCell(state, columns.state),
        ),
        width,
    )
}

fn padCell(value: &str, width: usize) -> String {
    let clipped = truncateStr(value, width);
    let len = clipped.chars().count();
    if len >= width {
        clipped
    } else {
        format!("{clipped:<width$}")
    }
}

fn truncateStr(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else if max <= 3 {
        ".".repeat(max)
    } else {
        let suffix = "...";
        let keep = max - suffix.len();
        let mut out: String = s.chars().take(keep).collect();
        out.push_str(suffix);
        out
    }
}
