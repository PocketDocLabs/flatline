#![allow(non_snake_case)]

//! Model profile panel: compact UI for viewing and switching configured profiles.

use construct::config::{ConfigScope, ModelTier};
use construct::control::ModelStatus;
use construct::model_catalog::ModelCatalogEntry;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum PanelMode {
    Profiles,
    Discover,
    Config,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConfigField {
    Model,
    Context,
    ThinkingMode,
    ReasoningEffort,
    ReasoningSummary,
    CreateProfile,
    RenameProfile,
    DeleteProfile,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ThinkingMode {
    Off,
    Provider,
    Prompt,
}

enum ProfileEdit {
    Create {
        text: String,
        sourceProfile: String,
    },
    Rename {
        original: String,
        text: String,
    },
    Delete {
        profile: String,
    },
    Context {
        profile: String,
        text: String,
        maxContextWindow: Option<usize>,
    },
}

pub enum PanelAction {
    None,
    Close,
    Save {
        scope: ConfigScope,
        tier: ModelTier,
        profile: String,
    },
    Discover {
        provider: String,
    },
    SaveDiscoveredModel {
        scope: ConfigScope,
        profile: String,
        model: ModelCatalogEntry,
    },
    CreateProfile {
        scope: ConfigScope,
        profile: String,
        sourceProfile: String,
    },
    RenameProfile {
        scope: ConfigScope,
        oldProfile: String,
        newProfile: String,
    },
    DeleteProfile {
        scope: ConfigScope,
        profile: String,
    },
    SaveContext {
        scope: ConfigScope,
        profile: String,
        contextWindow: usize,
    },
    SaveThinking {
        scope: ConfigScope,
        profile: String,
        promptThinking: bool,
        reasoningEffort: Option<String>,
        reasoningSummary: Option<String>,
    },
}

pub struct ModelPanel {
    status: ModelStatus,
    mode: PanelMode,
    selectedTier: ModelTier,
    selectedProfile: usize,
    scrollOffset: usize,
    lastVisibleCount: usize,
    selectedScope: usize,
    catalogProvider: String,
    selectedCatalogModel: usize,
    catalogScrollOffset: usize,
    catalogLastVisibleCount: usize,
    catalogEntries: Vec<ModelCatalogEntry>,
    catalogLoading: bool,
    catalogError: Option<String>,
    profileEdit: Option<ProfileEdit>,
    selectedConfigField: usize,
    discoverReturnMode: PanelMode,
    notice: Option<String>,
}

impl ModelPanel {
    pub fn new(status: ModelStatus) -> Self {
        let selectedProfile = status
            .profiles
            .iter()
            .position(|p| p.name == status.heavyProfile)
            .unwrap_or(0);
        let selectedScope = status
            .scopes
            .iter()
            .position(|s| s.scope == status.saveScope)
            .unwrap_or(0);
        let catalogProvider = status
            .profiles
            .get(selectedProfile)
            .map(|p| p.provider.clone())
            .unwrap_or_else(|| "openrouter".to_string());
        Self {
            status,
            mode: PanelMode::Profiles,
            selectedTier: ModelTier::Heavy,
            selectedProfile,
            scrollOffset: 0,
            lastVisibleCount: 5,
            selectedScope,
            catalogProvider,
            selectedCatalogModel: 0,
            catalogScrollOffset: 0,
            catalogLastVisibleCount: 5,
            catalogEntries: Vec::new(),
            catalogLoading: false,
            catalogError: None,
            profileEdit: None,
            selectedConfigField: 0,
            discoverReturnMode: PanelMode::Profiles,
            notice: None,
        }
    }

    pub fn handleKey(&mut self, key: KeyEvent) -> PanelAction {
        if self.profileEdit.is_some() {
            return self.handleProfileEditKey(key);
        }
        if self.mode == PanelMode::Discover {
            return self.handleDiscoverKey(key);
        }
        if self.mode == PanelMode::Config {
            return self.handleConfigKey(key);
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Char('e') | KeyCode::Char('E') => {
                self.mode = PanelMode::Config;
                self.selectedConfigField = 0;
                self.notice = None;
                PanelAction::None
            }
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
            KeyCode::Char('[') => {
                self.cycleScope(-1);
                PanelAction::None
            }
            KeyCode::Char(']') => {
                self.cycleScope(1);
                PanelAction::None
            }
            KeyCode::Enter | KeyCode::Char('s') => {
                let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
                    return PanelAction::None;
                };
                let profile = profile.name.clone();
                let scope = self.selectedScope();
                self.setActiveProfile(self.selectedTier, profile.clone());
                self.notice = Some(format!(
                    "{} -> {} ({})",
                    tierLabel(self.selectedTier),
                    profile,
                    scope.shortLabel()
                ));
                PanelAction::Save {
                    scope,
                    tier: self.selectedTier,
                    profile,
                }
            }
            _ => PanelAction::None,
        }
    }

    fn handleDiscoverKey(&mut self, key: KeyEvent) -> PanelAction {
        match key.code {
            KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Esc | KeyCode::Char('d') | KeyCode::Char('D') => {
                self.mode = self.discoverReturnMode;
                PanelAction::None
            }
            KeyCode::Char('p') => self.cycleCatalogProvider(1),
            KeyCode::Char('P') => self.cycleCatalogProvider(-1),
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selectedCatalogModel > 0 {
                    self.selectedCatalogModel -= 1;
                    self.adjustCatalogScroll();
                }
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selectedCatalogModel + 1 < self.catalogEntries.len() {
                    self.selectedCatalogModel += 1;
                    self.adjustCatalogScroll();
                }
                PanelAction::None
            }
            KeyCode::Char('[') => {
                self.cycleScope(-1);
                PanelAction::None
            }
            KeyCode::Char(']') => {
                self.cycleScope(1);
                PanelAction::None
            }
            KeyCode::Enter | KeyCode::Char('s') => {
                let Some(model) = self.catalogEntries.get(self.selectedCatalogModel).cloned()
                else {
                    return PanelAction::None;
                };
                let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
                    return PanelAction::None;
                };
                let profileName = profile.name.clone();
                let scope = self.selectedScope();
                self.applyDiscoveredModel(&profileName, &model);
                self.notice = Some(format!(
                    "{} -> {} ({})",
                    profileName,
                    model.id,
                    scope.shortLabel()
                ));
                PanelAction::SaveDiscoveredModel {
                    scope,
                    profile: profileName,
                    model,
                }
            }
            _ => PanelAction::None,
        }
    }

    fn handleConfigKey(&mut self, key: KeyEvent) -> PanelAction {
        let fields = configFields();
        match key.code {
            KeyCode::Char('q') => PanelAction::Close,
            KeyCode::Esc | KeyCode::Char('b') | KeyCode::Char('B') => {
                self.mode = PanelMode::Profiles;
                PanelAction::None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selectedConfigField > 0 {
                    self.selectedConfigField -= 1;
                }
                PanelAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selectedConfigField + 1 < fields.len() {
                    self.selectedConfigField += 1;
                }
                PanelAction::None
            }
            KeyCode::Char('[') => {
                self.cycleScope(-1);
                PanelAction::None
            }
            KeyCode::Char(']') => {
                self.cycleScope(1);
                PanelAction::None
            }
            KeyCode::Char(' ') => self.cycleSelectedConfigField(),
            KeyCode::Enter => self.editSelectedConfigField(),
            _ => PanelAction::None,
        }
    }

    fn handleProfileEditKey(&mut self, key: KeyEvent) -> PanelAction {
        match self.profileEdit.take() {
            Some(ProfileEdit::Create {
                mut text,
                sourceProfile,
            }) => match key.code {
                KeyCode::Esc => {
                    self.notice = Some("Create canceled".to_string());
                    PanelAction::None
                }
                KeyCode::Enter => self.finishCreateProfile(text, sourceProfile),
                KeyCode::Backspace => {
                    text.pop();
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Create {
                        text,
                        sourceProfile,
                    });
                    PanelAction::None
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    text.clear();
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Create {
                        text,
                        sourceProfile,
                    });
                    PanelAction::None
                }
                KeyCode::Char(ch) if validProfileNameChar(ch) => {
                    text.push(ch);
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Create {
                        text,
                        sourceProfile,
                    });
                    PanelAction::None
                }
                _ => {
                    self.profileEdit = Some(ProfileEdit::Create {
                        text,
                        sourceProfile,
                    });
                    PanelAction::None
                }
            },
            Some(ProfileEdit::Rename { original, mut text }) => match key.code {
                KeyCode::Esc => {
                    self.notice = Some("Rename canceled".to_string());
                    PanelAction::None
                }
                KeyCode::Enter => self.finishRenameProfile(original, text),
                KeyCode::Backspace => {
                    text.pop();
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Rename { original, text });
                    PanelAction::None
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    text.clear();
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Rename { original, text });
                    PanelAction::None
                }
                KeyCode::Char(ch) if validProfileNameChar(ch) => {
                    text.push(ch);
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Rename { original, text });
                    PanelAction::None
                }
                _ => {
                    self.profileEdit = Some(ProfileEdit::Rename { original, text });
                    PanelAction::None
                }
            },
            Some(ProfileEdit::Delete { profile }) => match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    self.finishDeleteProfile(profile)
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.notice = Some("Delete canceled".to_string());
                    PanelAction::None
                }
                _ => {
                    self.profileEdit = Some(ProfileEdit::Delete { profile });
                    PanelAction::None
                }
            },
            Some(ProfileEdit::Context {
                profile,
                mut text,
                maxContextWindow,
            }) => match key.code {
                KeyCode::Esc => {
                    self.notice = Some("Context edit canceled".to_string());
                    PanelAction::None
                }
                KeyCode::Enter => self.finishContextEdit(profile, text, maxContextWindow),
                KeyCode::Backspace => {
                    text.pop();
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Context {
                        profile,
                        text,
                        maxContextWindow,
                    });
                    PanelAction::None
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    text.clear();
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Context {
                        profile,
                        text,
                        maxContextWindow,
                    });
                    PanelAction::None
                }
                KeyCode::Char(ch) if validContextInputChar(ch) => {
                    text.push(ch);
                    self.notice = None;
                    self.profileEdit = Some(ProfileEdit::Context {
                        profile,
                        text,
                        maxContextWindow,
                    });
                    PanelAction::None
                }
                _ => {
                    self.profileEdit = Some(ProfileEdit::Context {
                        profile,
                        text,
                        maxContextWindow,
                    });
                    PanelAction::None
                }
            },
            None => PanelAction::None,
        }
    }

    pub fn setCatalogResult(
        &mut self,
        provider: String,
        result: std::result::Result<Vec<ModelCatalogEntry>, String>,
    ) {
        if provider != self.catalogProvider {
            return;
        }
        self.catalogLoading = false;
        match result {
            Ok(mut entries) => {
                if provider != "openai-codex" {
                    entries.sort_by(|a, b| a.id.cmp(&b.id));
                }
                self.catalogEntries = entries;
                self.catalogError = None;
                self.selectedCatalogModel = 0;
                self.catalogScrollOffset = 0;
                self.notice = Some(format!(
                    "{} models discovered for {}",
                    self.catalogEntries.len(),
                    self.catalogProvider
                ));
            }
            Err(error) => {
                self.catalogEntries.clear();
                self.catalogError = Some(error);
            }
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
        match self.mode {
            PanelMode::Profiles => self.renderProfiles(buf, inner, w, &mut y),
            PanelMode::Discover => self.renderDiscover(buf, inner, w, &mut y),
            PanelMode::Config => self.renderConfig(buf, inner, w, &mut y),
        }
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

        let rowCount = match self.mode {
            PanelMode::Profiles => self.status.profiles.len(),
            PanelMode::Discover => self.catalogEntries.len().max(3),
            PanelMode::Config => configFields().len().max(4),
        };
        let desiredInner = HEADER_LINES
            .saturating_add(PROFILE_HEADER_LINES)
            .saturating_add(rowCount as u16)
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
        width = width.max(self.widestSaveLineWidth());
        width = width.max(self.profileTableNaturalWidth());
        width = width.max(self.catalogTableNaturalWidth());
        width.max(self.configNaturalWidth())
    }

    fn startDiscover(&mut self) -> PanelAction {
        self.discoverReturnMode = self.mode;
        self.mode = PanelMode::Discover;
        if let Some(profile) = self.status.profiles.get(self.selectedProfile) {
            self.catalogProvider = profile.provider.clone();
        }
        self.catalogEntries.clear();
        self.catalogError = None;
        self.catalogLoading = true;
        self.selectedCatalogModel = 0;
        self.catalogScrollOffset = 0;
        self.notice = Some(format!("Discovering {} models", self.catalogProvider));
        PanelAction::Discover {
            provider: self.catalogProvider.clone(),
        }
    }

    fn cycleCatalogProvider(&mut self, direction: isize) -> PanelAction {
        let providers = self.providerChoices();
        if providers.is_empty() {
            return PanelAction::None;
        }
        let current = providers
            .iter()
            .position(|p| p == &self.catalogProvider)
            .unwrap_or(0) as isize;
        let next = (current + direction).rem_euclid(providers.len() as isize) as usize;
        self.catalogProvider = providers[next].clone();
        self.catalogEntries.clear();
        self.catalogError = None;
        self.catalogLoading = true;
        self.selectedCatalogModel = 0;
        self.catalogScrollOffset = 0;
        self.notice = Some(format!("Discovering {} models", self.catalogProvider));
        PanelAction::Discover {
            provider: self.catalogProvider.clone(),
        }
    }

    fn providerChoices(&self) -> Vec<String> {
        let providers = vec![
            "openrouter".to_string(),
            "deepseek".to_string(),
            "openai".to_string(),
            "openai-codex".to_string(),
        ];
        providers
    }

    fn startCreateProfile(&mut self) {
        let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return;
        };
        let sourceProfile = profile.name.clone();
        let text = self.uniqueProfileName(&format!("{sourceProfile}Copy"));
        self.profileEdit = Some(ProfileEdit::Create {
            text,
            sourceProfile,
        });
        self.notice = None;
    }

    fn startRenameProfile(&mut self) {
        let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return;
        };
        self.profileEdit = Some(ProfileEdit::Rename {
            original: profile.name.clone(),
            text: profile.name.clone(),
        });
        self.notice = None;
    }

    fn startDeleteProfile(&mut self) {
        let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return;
        };
        if self.profileIsAssigned(&profile.name) {
            self.notice = Some(format!("Switch tiers before deleting {}", profile.name));
            return;
        }
        self.profileEdit = Some(ProfileEdit::Delete {
            profile: profile.name.clone(),
        });
        self.notice = None;
    }

    fn startContextEdit(&mut self) {
        let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return;
        };
        self.profileEdit = Some(ProfileEdit::Context {
            profile: profile.name.clone(),
            text: formatContextInput(profile.contextWindow),
            maxContextWindow: profile.maxContextWindow,
        });
        self.notice = None;
    }

    fn finishCreateProfile(&mut self, profileName: String, sourceProfile: String) -> PanelAction {
        if let Err(message) = self.validateProfileName(&profileName) {
            self.notice = Some(message);
            self.profileEdit = Some(ProfileEdit::Create {
                text: profileName,
                sourceProfile,
            });
            return PanelAction::None;
        }
        let Some(source) = self
            .status
            .profiles
            .iter()
            .find(|profile| profile.name == sourceProfile)
            .cloned()
        else {
            self.notice = Some(format!("Source profile {sourceProfile} is gone"));
            return PanelAction::None;
        };

        let scope = self.selectedScope();
        let mut next = source;
        next.name = profileName.clone();
        let insertAt = self.selectedProfile.saturating_add(1);
        self.status.profiles.insert(insertAt, next);
        self.selectedProfile = insertAt;
        self.adjustScroll();
        self.notice = Some(format!("Created {profileName} ({})", scope.shortLabel()));
        PanelAction::CreateProfile {
            scope,
            profile: profileName,
            sourceProfile,
        }
    }

    fn finishRenameProfile(&mut self, oldProfile: String, newProfile: String) -> PanelAction {
        if oldProfile == newProfile {
            self.notice = Some("Profile name unchanged".to_string());
            return PanelAction::None;
        }
        if let Err(message) = self.validateProfileName(&newProfile) {
            self.notice = Some(message);
            self.profileEdit = Some(ProfileEdit::Rename {
                original: oldProfile,
                text: newProfile,
            });
            return PanelAction::None;
        }

        let scope = self.selectedScope();
        if let Some(profile) = self
            .status
            .profiles
            .iter_mut()
            .find(|profile| profile.name == oldProfile)
        {
            profile.name = newProfile.clone();
        }
        if self.status.heavyProfile == oldProfile {
            self.status.heavyProfile = newProfile.clone();
        }
        if self.status.lightProfile == oldProfile {
            self.status.lightProfile = newProfile.clone();
        }
        if self.status.utilityProfile == oldProfile {
            self.status.utilityProfile = newProfile.clone();
        }
        self.notice = Some(format!(
            "Renamed {oldProfile} -> {newProfile} ({})",
            scope.shortLabel()
        ));
        PanelAction::RenameProfile {
            scope,
            oldProfile,
            newProfile,
        }
    }

    fn finishDeleteProfile(&mut self, profileName: String) -> PanelAction {
        if self.profileIsAssigned(&profileName) {
            self.notice = Some(format!("Switch tiers before deleting {profileName}"));
            return PanelAction::None;
        }
        let Some(idx) = self
            .status
            .profiles
            .iter()
            .position(|profile| profile.name == profileName)
        else {
            self.notice = Some(format!("Profile {profileName} is gone"));
            return PanelAction::None;
        };
        let scope = self.selectedScope();
        self.status.profiles.remove(idx);
        if self.status.profiles.is_empty() {
            self.selectedProfile = 0;
            self.scrollOffset = 0;
        } else {
            self.selectedProfile = idx.min(self.status.profiles.len() - 1);
            self.adjustScroll();
        }
        self.notice = Some(format!("Deleted {profileName} ({})", scope.shortLabel()));
        PanelAction::DeleteProfile {
            scope,
            profile: profileName,
        }
    }

    fn finishContextEdit(
        &mut self,
        profileName: String,
        text: String,
        maxContextWindow: Option<usize>,
    ) -> PanelAction {
        let contextWindow = match parseContextWindowInput(&text) {
            Ok(contextWindow) => contextWindow,
            Err(message) => {
                self.notice = Some(message);
                self.profileEdit = Some(ProfileEdit::Context {
                    profile: profileName,
                    text,
                    maxContextWindow,
                });
                return PanelAction::None;
            }
        };
        if let Some(max) = maxContextWindow
            && contextWindow > max
        {
            self.notice = Some(format!("Context must be <= {}", formatContextInput(max)));
            self.profileEdit = Some(ProfileEdit::Context {
                profile: profileName,
                text,
                maxContextWindow,
            });
            return PanelAction::None;
        }

        let Some(profile) = self
            .status
            .profiles
            .iter_mut()
            .find(|profile| profile.name == profileName)
        else {
            self.notice = Some(format!("Profile {profileName} is gone"));
            return PanelAction::None;
        };
        profile.contextWindow = contextWindow;
        let scope = self.selectedScope();
        self.notice = Some(format!(
            "Context for {profileName}: {} ({})",
            formatContextInput(contextWindow),
            scope.shortLabel()
        ));
        PanelAction::SaveContext {
            scope,
            profile: profileName,
            contextWindow,
        }
    }

    fn editSelectedConfigField(&mut self) -> PanelAction {
        match self.selectedConfigField() {
            ConfigField::Model => self.startDiscover(),
            ConfigField::Context => {
                self.startContextEdit();
                PanelAction::None
            }
            ConfigField::ThinkingMode => self.cycleThinkingMode(),
            ConfigField::ReasoningEffort => {
                let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
                    self.notice = Some("No profile selected".to_string());
                    return PanelAction::None;
                };
                if thinkingMode(profile) != ThinkingMode::Provider {
                    self.notice = Some("Set thinking mode to provider first".to_string());
                    return PanelAction::None;
                }
                self.cycleProviderEffort()
            }
            ConfigField::ReasoningSummary => {
                let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
                    self.notice = Some("No profile selected".to_string());
                    return PanelAction::None;
                };
                if thinkingMode(profile) != ThinkingMode::Provider {
                    self.notice = Some("Set thinking mode to provider first".to_string());
                    return PanelAction::None;
                }
                self.cycleReasoningSummary()
            }
            ConfigField::CreateProfile => {
                self.startCreateProfile();
                PanelAction::None
            }
            ConfigField::RenameProfile => {
                self.startRenameProfile();
                PanelAction::None
            }
            ConfigField::DeleteProfile => {
                self.startDeleteProfile();
                PanelAction::None
            }
        }
    }

    fn cycleSelectedConfigField(&mut self) -> PanelAction {
        match self.selectedConfigField() {
            ConfigField::Model => self.startDiscover(),
            ConfigField::Context => {
                self.startContextEdit();
                PanelAction::None
            }
            ConfigField::ThinkingMode => self.cycleThinkingMode(),
            ConfigField::ReasoningEffort => {
                let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
                    self.notice = Some("No profile selected".to_string());
                    return PanelAction::None;
                };
                if thinkingMode(profile) != ThinkingMode::Provider {
                    self.notice = Some("Set thinking mode to provider first".to_string());
                    return PanelAction::None;
                }
                self.cycleProviderEffort()
            }
            ConfigField::ReasoningSummary => {
                let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
                    self.notice = Some("No profile selected".to_string());
                    return PanelAction::None;
                };
                if thinkingMode(profile) != ThinkingMode::Provider {
                    self.notice = Some("Set thinking mode to provider first".to_string());
                    return PanelAction::None;
                }
                self.cycleReasoningSummary()
            }
            ConfigField::CreateProfile => {
                self.startCreateProfile();
                PanelAction::None
            }
            ConfigField::RenameProfile => {
                self.startRenameProfile();
                PanelAction::None
            }
            ConfigField::DeleteProfile => {
                self.startDeleteProfile();
                PanelAction::None
            }
        }
    }

    fn cycleThinkingMode(&mut self) -> PanelAction {
        let Some(profile) = self.status.profiles.get_mut(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return PanelAction::None;
        };
        match thinkingMode(profile) {
            ThinkingMode::Off => {
                let Some(effort) = firstProviderEffort(profile) else {
                    self.notice =
                        Some("Provider thinking is unavailable for this model".to_string());
                    return PanelAction::None;
                };
                profile.promptThinking = false;
                profile.reasoningEffort = Some(effort);
            }
            ThinkingMode::Provider => {
                profile.promptThinking = true;
            }
            ThinkingMode::Prompt => {
                profile.promptThinking = false;
                profile.reasoningEffort = None;
                profile.reasoningSummary = None;
            }
        }
        self.saveThinkingForSelected()
    }

    fn cycleProviderEffort(&mut self) -> PanelAction {
        let Some(profile) = self.status.profiles.get_mut(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return PanelAction::None;
        };
        let next = cycleProviderEffort(profile);
        let Some(next) = next else {
            self.notice = Some("Provider thinking is unavailable for this model".to_string());
            return PanelAction::None;
        };
        profile.reasoningEffort = Some(next);
        self.saveThinkingForSelected()
    }

    fn cycleReasoningSummary(&mut self) -> PanelAction {
        let Some(profile) = self.status.profiles.get_mut(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return PanelAction::None;
        };
        profile.reasoningSummary = cycleOptionalValue(
            profile.reasoningSummary.as_deref(),
            reasoningSummaryChoices(),
        );
        self.saveThinkingForSelected()
    }

    fn saveThinkingForSelected(&mut self) -> PanelAction {
        let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
            self.notice = Some("No profile selected".to_string());
            return PanelAction::None;
        };
        let scope = self.selectedScope();
        let profileName = profile.name.clone();
        let promptThinking = profile.promptThinking;
        let reasoningEffort = profile.reasoningEffort.clone();
        let reasoningSummary = profile.reasoningSummary.clone();
        self.notice = Some(format!(
            "Thinking for {profileName}: {} ({})",
            thinkingModeLabel(thinkingMode(profile)),
            scope.shortLabel()
        ));
        PanelAction::SaveThinking {
            scope,
            profile: profileName,
            promptThinking,
            reasoningEffort,
            reasoningSummary,
        }
    }

    fn uniqueProfileName(&self, base: &str) -> String {
        let seed = if base.is_empty() { "profile" } else { base };
        if !self.profileNameExists(seed) {
            return seed.to_string();
        }
        for idx in 2..1000 {
            let candidate = format!("{seed}{idx}");
            if !self.profileNameExists(&candidate) {
                return candidate;
            }
        }
        format!("{seed}1000")
    }

    fn validateProfileName(&self, profileName: &str) -> Result<(), String> {
        if profileName.is_empty() {
            return Err("Profile name cannot be empty".to_string());
        }
        if self.profileNameExists(profileName) {
            return Err(format!("Profile {profileName} already exists"));
        }
        if !profileName.chars().all(validProfileNameChar) {
            return Err("Use letters, numbers, dot, dash, or underscore".to_string());
        }
        Ok(())
    }

    fn profileNameExists(&self, profileName: &str) -> bool {
        self.status
            .profiles
            .iter()
            .any(|profile| profile.name == profileName)
    }

    fn profileIsAssigned(&self, profileName: &str) -> bool {
        self.status.heavyProfile == profileName
            || self.status.lightProfile == profileName
            || self.status.utilityProfile == profileName
    }

    fn setTier(&mut self, tier: ModelTier) {
        self.selectedTier = tier;
        let active = self.activeProfileFor(tier).to_string();
        if let Some(idx) = self.status.profiles.iter().position(|p| p.name == active) {
            self.selectedProfile = idx;
            self.adjustScroll();
        }
    }

    fn selectedScope(&self) -> ConfigScope {
        self.status
            .scopes
            .get(self.selectedScope)
            .map(|s| s.scope)
            .unwrap_or(self.status.saveScope)
    }

    fn selectedConfigField(&self) -> ConfigField {
        configFields()
            .get(self.selectedConfigField)
            .copied()
            .unwrap_or(ConfigField::Context)
    }

    fn selectedScopeStatus(&self) -> Option<&construct::control::ModelConfigScopeStatus> {
        self.status.scopes.get(self.selectedScope)
    }

    fn cycleScope(&mut self, direction: isize) {
        if self.status.scopes.is_empty() {
            return;
        }
        let len = self.status.scopes.len() as isize;
        self.selectedScope = (self.selectedScope as isize + direction).rem_euclid(len) as usize;
        self.status.saveScope = self.selectedScope();
        if let Some(scope) = self.selectedScopeStatus() {
            let label = scope.label.clone();
            let path = scope.path.clone();
            self.status.configPath = path;
            self.notice = Some(format!("Save target: {label}"));
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

    fn applyDiscoveredModel(&mut self, profileName: &str, model: &ModelCatalogEntry) {
        if let Some(profile) = self
            .status
            .profiles
            .iter_mut()
            .find(|profile| profile.name == profileName)
        {
            profile.provider = model.provider.clone();
            profile.model = model.id.clone();
            if let Some(contextWindow) = model.contextWindow {
                profile.contextWindow = contextWindow;
                profile.maxContextWindow = Some(contextWindow);
            }
            if let Some(effort) = &model.defaultReasoningEffort {
                profile.reasoningEffort = Some(effort.clone());
            }
            profile.reasoningEfforts = model.reasoningEfforts.clone();
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
        if let Some(edit) = &self.profileEdit {
            let prompt = match edit {
                ProfileEdit::Create {
                    text,
                    sourceProfile,
                } => format!(" New profile from {sourceProfile}: {text}"),
                ProfileEdit::Rename { original, text } => {
                    format!(" Rename {original} to: {text}")
                }
                ProfileEdit::Delete { profile } => {
                    format!(
                        " Delete {profile} from {}? y confirm / n cancel",
                        self.selectedScope().shortLabel()
                    )
                }
                ProfileEdit::Context {
                    profile,
                    text,
                    maxContextWindow,
                } => {
                    let max = maxContextWindow
                        .map(formatContextInput)
                        .unwrap_or_else(|| "unknown".to_string());
                    format!(" Context for {profile} (max {max}): {text}")
                }
            };
            if let Some(notice) = &self.notice {
                format!(" {notice}.{}", prompt)
            } else {
                prompt
            }
        } else if let Some(notice) = &self.notice {
            format!(" Saved: {notice}")
        } else if self.mode == PanelMode::Discover {
            let profile = self
                .status
                .profiles
                .get(self.selectedProfile)
                .map(|p| p.name.as_str())
                .unwrap_or("profile");
            format!(
                " Discover {} models. Enter applies to profile {profile}.",
                self.catalogProvider
            )
        } else if self.mode == PanelMode::Config {
            let profile = self
                .status
                .profiles
                .get(self.selectedProfile)
                .map(|p| p.name.as_str())
                .unwrap_or("profile");
            format!(" Configuring {profile}. Space cycles, Enter edits.")
        } else {
            format!(
                " Editing {} tier. Enter saves and leaves this panel open.",
                tierLabel(self.selectedTier)
            )
        }
    }

    fn saveLine(&self) -> String {
        if let Some(scope) = self.selectedScopeStatus() {
            saveLineFor(&scope.label, &scope.path)
        } else {
            format!(" Save to {}", self.status.configPath)
        }
    }

    fn widestSaveLineWidth(&self) -> usize {
        self.status
            .scopes
            .iter()
            .map(|scope| saveLineFor(&scope.label, &scope.path).chars().count())
            .max()
            .unwrap_or_else(|| self.saveLine().chars().count())
    }

    fn footerItems(&self) -> Vec<String> {
        if let Some(edit) = &self.profileEdit {
            return match edit {
                ProfileEdit::Delete { .. } => {
                    vec!["y delete".to_string(), "n/esc cancel".to_string()]
                }
                ProfileEdit::Create { .. } | ProfileEdit::Rename { .. } => {
                    vec![
                        "type profile name".to_string(),
                        "ctrl-u clear".to_string(),
                        "enter save".to_string(),
                        "esc cancel".to_string(),
                    ]
                }
                ProfileEdit::Context { .. } => vec![
                    "type tokens or 128k".to_string(),
                    "ctrl-u clear".to_string(),
                    "enter save".to_string(),
                    "esc cancel".to_string(),
                ],
            };
        }
        match self.mode {
            PanelMode::Profiles => vec![
                format!("h/l/u tier: {}", tierLabel(self.selectedTier)),
                "up/down profile".to_string(),
                "e config".to_string(),
                "[/] target".to_string(),
                "enter save".to_string(),
                "q close".to_string(),
            ],
            PanelMode::Discover => vec![
                format!("provider: {}", self.catalogProvider),
                "p/P provider".to_string(),
                "up/down model".to_string(),
                "[/] save target".to_string(),
                "enter apply".to_string(),
                "esc back".to_string(),
                "q close".to_string(),
            ],
            PanelMode::Config => vec![
                "up/down field".to_string(),
                "space cycle".to_string(),
                "enter edit".to_string(),
                "[/] target".to_string(),
                "esc back".to_string(),
                "q close".to_string(),
            ],
        }
    }

    fn footerLines(&self, width: usize) -> [String; 2] {
        let items = self.footerItems();
        if items.is_empty() {
            return [String::new(), String::new()];
        }

        let full = footerLineFromItems(&items);
        if full.chars().count() <= width {
            return [full, String::new()];
        }

        let bestSplit = greedyFooterSplit(&items, width);

        [
            footerLineFromItems(&items[..bestSplit]),
            footerLineFromItems(&items[bestSplit..]),
        ]
    }

    fn profileTableNaturalWidth(&self) -> usize {
        let mut width = profileNaturalWidth("  ", "profile", "provider", "model", "ctx", "state");
        for profile in &self.status.profiles {
            let state = if profile.configured {
                "ready"
            } else {
                "needs auth"
            };
            let ctx = profileContextLabel(profile.contextWindow, profile.maxContextWindow);
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

    fn catalogTableNaturalWidth(&self) -> usize {
        let mut width = catalogNaturalWidth("  ", "model", "name", "ctx", "effort", "pricing");
        for model in &self.catalogEntries {
            width = width.max(catalogNaturalWidth(
                "> ",
                &model.id,
                &model.name,
                &contextLabel(model.contextWindow),
                &effortLabel(model),
                &pricingLabel(model),
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
            columns.context = columns.context.max(
                profileContextLabel(profile.contextWindow, profile.maxContextWindow)
                    .chars()
                    .count(),
            );
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
        let selectionStyle = if self.profileEdit.is_some() && self.notice.is_some() {
            style(FG_WARN, BG)
        } else if self.notice.is_some() {
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
            &profileRow(
                columns, "  ", "profile", "provider", "model", "ctx", "state", w,
            ),
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
            let ctx = profileContextLabel(profile.contextWindow, profile.maxContextWindow);
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

    fn renderDiscover(&mut self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        let columns = catalogColumns(&self.catalogEntries, w);
        line(
            buf,
            inner.x,
            *y,
            w,
            &catalogRow(
                columns, "  ", "model", "name", "ctx", "effort", "pricing", w,
            ),
            style(FG_DIM, BG),
        );
        *y += 1;

        let available = (inner.y + inner.height).saturating_sub(*y + FOOTER_RESERVE) as usize;
        self.catalogLastVisibleCount = available.max(1);

        if self.catalogLoading {
            line(
                buf,
                inner.x,
                *y,
                w,
                &truncateStr(&format!(" Loading {} models...", self.catalogProvider), w),
                style(FG_DIM, BG),
            );
            *y += 1;
        } else if let Some(error) = &self.catalogError {
            line(
                buf,
                inner.x,
                *y,
                w,
                &truncateStr(&format!(" Discovery failed: {error}"), w),
                style(FG_WARN, BG),
            );
            *y += 1;
        } else if self.catalogEntries.is_empty() {
            line(
                buf,
                inner.x,
                *y,
                w,
                &truncateStr(" No models found.", w),
                style(FG_DIM, BG),
            );
            *y += 1;
        } else {
            let visibleCount = available.min(
                self.catalogEntries
                    .len()
                    .saturating_sub(self.catalogScrollOffset),
            );
            self.catalogLastVisibleCount = visibleCount.max(1);
            for i in 0..visibleCount {
                let idx = self.catalogScrollOffset + i;
                let Some(model) = self.catalogEntries.get(idx) else {
                    break;
                };
                let selected = idx == self.selectedCatalogModel;
                let bg = if selected { BG_SELECTED } else { BG };
                let marker = if selected { ">" } else { " " };
                let text = catalogRow(
                    columns,
                    marker,
                    &model.id,
                    &model.name,
                    &contextLabel(model.contextWindow),
                    &effortLabel(model),
                    &pricingLabel(model),
                    w,
                );
                line(buf, inner.x, *y, w, &text, style(FG_PRIMARY, bg));
                *y += 1;
            }
        }

        while *y < inner.y + inner.height - FOOTER_RESERVE {
            line(buf, inner.x, *y, w, "", style(FG_PRIMARY, BG));
            *y += 1;
        }
    }

    fn renderConfig(&mut self, buf: &mut Buffer, inner: Rect, w: usize, y: &mut u16) {
        let Some(profile) = self.status.profiles.get(self.selectedProfile) else {
            line(
                buf,
                inner.x,
                *y,
                w,
                &truncateStr(" No profile selected.", w),
                style(FG_DIM, BG),
            );
            return;
        };

        let title = format!(
            " Profile: {}   {} / {}",
            profile.name, profile.provider, profile.model
        );
        line(
            buf,
            inner.x,
            *y,
            w,
            &truncateStr(&title, w),
            style(FG_PRIMARY, BG),
        );
        *y += 1;

        let labelWidth = configLabelWidth();
        let fields = configFields();
        for (idx, field) in fields.iter().enumerate() {
            let selected = idx == self.selectedConfigField;
            let disabled = configFieldDisabled(profile, *field);
            let bg = if selected { BG_SELECTED } else { BG };
            let marker = if selected { ">" } else { " " };
            let label = configFieldLabel(*field);
            let value = self.configFieldValue(profile, *field);
            let row = format!(
                "{marker} {label:<labelWidth$}  {value}",
                labelWidth = labelWidth
            );
            let fg = if disabled {
                if selected { FG_DIM } else { FG_MUTED }
            } else {
                FG_PRIMARY
            };
            line(buf, inner.x, *y, w, &truncateStr(&row, w), style(fg, bg));
            *y += 1;
        }

        while *y < inner.y + inner.height - FOOTER_RESERVE {
            line(buf, inner.x, *y, w, "", style(FG_PRIMARY, BG));
            *y += 1;
        }
    }

    fn configNaturalWidth(&self) -> usize {
        let mut width = " Profile: ".chars().count();
        for profile in &self.status.profiles {
            width = width.max(
                format!(
                    " Profile: {}   {} / {}",
                    profile.name, profile.provider, profile.model
                )
                .chars()
                .count(),
            );
            for &field in configFields() {
                width = width.max(
                    format!(
                        "> {:<labelWidth$}  {}",
                        configFieldLabel(field),
                        self.configFieldValue(profile, field),
                        labelWidth = configLabelWidth()
                    )
                    .chars()
                    .count(),
                );
            }
        }
        width
    }

    fn configFieldValue(
        &self,
        profile: &construct::control::ModelProfileStatus,
        field: ConfigField,
    ) -> String {
        match field {
            ConfigField::Model => format!("{} / {}", profile.provider, profile.model),
            ConfigField::Context => {
                profileContextLabel(profile.contextWindow, profile.maxContextWindow)
            }
            ConfigField::ThinkingMode => thinkingModeLabel(thinkingMode(profile)).to_string(),
            ConfigField::ReasoningEffort => {
                if profile.reasoningEfforts.is_empty() {
                    "unavailable".to_string()
                } else {
                    optionalValueLabel(profile.reasoningEffort.as_deref()).to_string()
                }
            }
            ConfigField::ReasoningSummary => {
                optionalValueLabel(profile.reasoningSummary.as_deref()).to_string()
            }
            ConfigField::CreateProfile => "copy selected profile".to_string(),
            ConfigField::RenameProfile => profile.name.clone(),
            ConfigField::DeleteProfile => {
                if self.profileIsAssigned(&profile.name) {
                    "assigned to tier".to_string()
                } else {
                    "confirm before delete".to_string()
                }
            }
        }
    }

    fn renderFooter(&self, buf: &mut Buffer, popup: Rect, inner: Rect, w: usize) {
        let y = popup.y + popup.height.saturating_sub(3);
        let [first, second] = self.footerLines(w);
        line(
            buf,
            inner.x,
            y,
            w,
            &truncateStr(&first, w),
            style(FG_DIM, BG),
        );
        line(
            buf,
            inner.x,
            y.saturating_add(1),
            w,
            &truncateStr(&second, w),
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

    fn adjustCatalogScroll(&mut self) {
        if self.selectedCatalogModel < self.catalogScrollOffset {
            self.catalogScrollOffset = self.selectedCatalogModel;
        } else if self.selectedCatalogModel
            >= self.catalogScrollOffset + self.catalogLastVisibleCount
        {
            self.catalogScrollOffset = self
                .selectedCatalogModel
                .saturating_sub(self.catalogLastVisibleCount.saturating_sub(1));
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

fn saveLineFor(label: &str, path: &str) -> String {
    format!(" Save to {label}: {path}")
}

fn configFields() -> &'static [ConfigField] {
    &[
        ConfigField::Model,
        ConfigField::Context,
        ConfigField::ThinkingMode,
        ConfigField::ReasoningEffort,
        ConfigField::ReasoningSummary,
        ConfigField::CreateProfile,
        ConfigField::RenameProfile,
        ConfigField::DeleteProfile,
    ]
}

fn configFieldLabel(field: ConfigField) -> &'static str {
    match field {
        ConfigField::Model => "model",
        ConfigField::Context => "context",
        ConfigField::ThinkingMode => "thinking mode",
        ConfigField::ReasoningEffort => "provider effort",
        ConfigField::ReasoningSummary => "reasoning summary",
        ConfigField::CreateProfile => "new profile",
        ConfigField::RenameProfile => "rename profile",
        ConfigField::DeleteProfile => "delete profile",
    }
}

fn configLabelWidth() -> usize {
    configFields()
        .iter()
        .map(|field| configFieldLabel(*field).chars().count())
        .max()
        .unwrap_or(0)
}

fn reasoningSummaryChoices() -> &'static [&'static str] {
    &["auto", "concise", "detailed"]
}

fn cycleOptionalValue(current: Option<&str>, choices: &[&str]) -> Option<String> {
    let Some(current) = current else {
        return choices.first().map(|value| (*value).to_string());
    };
    let idx = choices.iter().position(|value| *value == current)?;
    choices.get(idx + 1).map(|value| (*value).to_string())
}

fn optionalValueLabel(value: Option<&str>) -> &str {
    value.unwrap_or("off")
}

fn configFieldDisabled(
    profile: &construct::control::ModelProfileStatus,
    field: ConfigField,
) -> bool {
    matches!(
        field,
        ConfigField::ReasoningEffort | ConfigField::ReasoningSummary
    ) && thinkingMode(profile) != ThinkingMode::Provider
}

fn thinkingMode(profile: &construct::control::ModelProfileStatus) -> ThinkingMode {
    if profile.promptThinking {
        ThinkingMode::Prompt
    } else if profile.reasoningEffort.is_some() || profile.reasoningSummary.is_some() {
        ThinkingMode::Provider
    } else {
        ThinkingMode::Off
    }
}

fn thinkingModeLabel(mode: ThinkingMode) -> &'static str {
    match mode {
        ThinkingMode::Off => "off",
        ThinkingMode::Provider => "provider",
        ThinkingMode::Prompt => "prompt scratchpad",
    }
}

fn firstProviderEffort(profile: &construct::control::ModelProfileStatus) -> Option<String> {
    if profile.reasoningEfforts.is_empty() {
        None
    } else if let Some(effort) = &profile.reasoningEffort
        && profile
            .reasoningEfforts
            .iter()
            .any(|candidate| candidate == effort)
    {
        Some(effort.clone())
    } else if profile
        .reasoningEfforts
        .iter()
        .any(|effort| effort == "medium")
    {
        Some("medium".to_string())
    } else {
        profile.reasoningEfforts.first().cloned()
    }
}

fn cycleProviderEffort(profile: &construct::control::ModelProfileStatus) -> Option<String> {
    let efforts = &profile.reasoningEfforts;
    if efforts.is_empty() {
        return None;
    }
    let current = profile.reasoningEffort.as_deref();
    let idx = current
        .and_then(|value| efforts.iter().position(|effort| effort == value))
        .unwrap_or(efforts.len().saturating_sub(1));
    efforts.get((idx + 1) % efforts.len()).cloned()
}

fn footerLineFromItems(items: &[String]) -> String {
    if items.is_empty() {
        String::new()
    } else {
        format!(" {} ", items.join("   "))
    }
}

fn greedyFooterSplit(items: &[String], width: usize) -> usize {
    if items.len() <= 1 {
        return items.len();
    }

    let mut split = 1;
    for candidate in 1..items.len() {
        let lineWidth = footerLineFromItems(&items[..=candidate]).chars().count();
        if lineWidth > width {
            break;
        }
        split = candidate + 1;
    }
    split.min(items.len() - 1)
}

fn tierLabel(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Heavy => "heavy",
        ModelTier::Light => "light",
        ModelTier::Utility => "utility",
    }
}

fn contextLabel(contextWindow: Option<usize>) -> String {
    contextWindow
        .map(|ctx| format!("{}k", ctx / 1000))
        .unwrap_or_else(|| "-".to_string())
}

fn profileContextLabel(contextWindow: usize, maxContextWindow: Option<usize>) -> String {
    let current = formatContextInput(contextWindow);
    match maxContextWindow {
        Some(max) if contextWindow < max => format!("{current}/{}", formatContextInput(max)),
        _ => current,
    }
}

fn formatContextInput(contextWindow: usize) -> String {
    if contextWindow.is_multiple_of(1_000_000) {
        format!("{}m", contextWindow / 1_000_000)
    } else if contextWindow.is_multiple_of(100_000) && contextWindow >= 1_000_000 {
        let whole = contextWindow / 1_000_000;
        let tenth = (contextWindow % 1_000_000) / 100_000;
        format!("{whole}.{tenth}m")
    } else if contextWindow.is_multiple_of(1000) {
        format!("{}k", contextWindow / 1000)
    } else {
        contextWindow.to_string()
    }
}

fn parseContextWindowInput(input: &str) -> Result<usize, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Context cannot be empty".to_string());
    }
    let (number, multiplier) = if let Some(number) = trimmed
        .strip_suffix('k')
        .or_else(|| trimmed.strip_suffix('K'))
    {
        (number, 1_000f64)
    } else if let Some(number) = trimmed
        .strip_suffix('m')
        .or_else(|| trimmed.strip_suffix('M'))
    {
        (number, 1_000_000f64)
    } else {
        (trimmed, 1f64)
    };
    let value = number
        .parse::<f64>()
        .map_err(|_| "Use a number like 128k or 128000".to_string())?;
    if !value.is_finite() || value <= 0.0 {
        return Err("Context must be greater than zero".to_string());
    }
    let tokens = (value * multiplier).round();
    if tokens > usize::MAX as f64 {
        return Err("Context is too large".to_string());
    }
    Ok(tokens as usize)
}

fn validContextInputChar(ch: char) -> bool {
    ch.is_ascii_digit() || matches!(ch, '.' | 'k' | 'K' | 'm' | 'M')
}

fn pricingLabel(model: &ModelCatalogEntry) -> String {
    match (&model.promptPrice, &model.completionPrice) {
        (Some(prompt), Some(completion)) => format!("{prompt}/{completion}"),
        _ => "-".to_string(),
    }
}

fn validProfileNameChar(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_')
}

fn effortLabel(model: &ModelCatalogEntry) -> String {
    if model.reasoningEfforts.is_empty() {
        return model
            .defaultReasoningEffort
            .clone()
            .unwrap_or_else(|| "-".to_string());
    }
    let efforts = model.reasoningEfforts.join("/");
    match &model.defaultReasoningEffort {
        Some(default) => format!("{default} [{efforts}]"),
        None => efforts,
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

#[derive(Clone, Copy)]
struct CatalogColumns {
    marker: usize,
    id: usize,
    name: usize,
    context: usize,
    effort: usize,
    pricing: usize,
}

impl CatalogColumns {
    fn total(self) -> usize {
        self.marker + self.id + self.name + self.context + self.effort + self.pricing + 5
    }

    fn fitTo(mut self, width: usize, min: CatalogColumns) -> CatalogColumns {
        while self.total() > width && self.name > min.name {
            self.name -= 1;
        }
        while self.total() > width && self.id > min.id {
            self.id -= 1;
        }
        while self.total() > width && self.pricing > min.pricing {
            self.pricing -= 1;
        }
        while self.total() > width && self.effort > min.effort {
            self.effort -= 1;
        }

        let mut extra = width.saturating_sub(self.total());
        let idTarget = self.id.max(38);
        let idExtra = extra.min(idTarget.saturating_sub(self.id));
        self.id += idExtra;
        extra -= idExtra;
        let effortTarget = self.effort.max(30);
        let effortExtra = extra.min(effortTarget.saturating_sub(self.effort));
        self.effort += effortExtra;
        extra -= effortExtra;
        self.name += extra;
        self
    }
}

fn catalogColumns(models: &[ModelCatalogEntry], width: usize) -> CatalogColumns {
    let mut columns = CatalogColumns {
        marker: 2,
        id: "model".chars().count(),
        name: "name".chars().count(),
        context: "ctx".chars().count().max(7),
        effort: "effort".chars().count().max(10),
        pricing: "pricing".chars().count().max(12),
    };
    for model in models {
        columns.id = columns.id.max(model.id.chars().count());
        columns.name = columns.name.max(model.name.chars().count());
        columns.context = columns
            .context
            .max(contextLabel(model.contextWindow).chars().count());
        columns.effort = columns.effort.max(effortLabel(model).chars().count());
        columns.pricing = columns.pricing.max(pricingLabel(model).chars().count());
    }
    columns.fitTo(
        width,
        CatalogColumns {
            marker: 2,
            id: 16,
            name: 12,
            context: 7,
            effort: 10,
            pricing: 12,
        },
    )
}

fn catalogNaturalWidth(
    marker: &str,
    id: &str,
    name: &str,
    context: &str,
    effort: &str,
    pricing: &str,
) -> usize {
    marker.chars().count()
        + id.chars().count()
        + name.chars().count()
        + context.chars().count()
        + effort.chars().count()
        + pricing.chars().count()
        + 5
}

#[allow(clippy::too_many_arguments)]
fn catalogRow(
    columns: CatalogColumns,
    marker: &str,
    id: &str,
    name: &str,
    context: &str,
    effort: &str,
    pricing: &str,
    width: usize,
) -> String {
    if width < columns.total() {
        return truncateStr(
            &format!("{marker} {id}  {name}  {context}  {effort}"),
            width,
        );
    }
    truncateStr(
        &format!(
            "{} {} {} {} {} {}",
            padCell(marker, columns.marker),
            padCell(id, columns.id),
            padCell(name, columns.name),
            padCell(context, columns.context),
            padCell(effort, columns.effort),
            padCell(pricing, columns.pricing),
        ),
        width,
    )
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

#[allow(clippy::too_many_arguments)]
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
