//! Named registry of shells, one PTY per entry.
//!
//! Replaces the single-shell model. Each session owns a `ShellRegistry`
//! holding a default `main` shell plus any agent- or user-spawned ones.
//! Tools that touch a shell (`shell`, `readTerminal`, `shellHistory`,
//! `readOutput`, `searchOutput`) target the active shell by default,
//! or a named one if given.
//!
//! Newly-spawned shells deliver their `ShellIo` to the harness through
//! the `ioTx` channel set at construction time. The harness (deck or
//! a headless drainer) receives `(name, ShellIo)` tuples and wires
//! them into rendering / output drainage.
//!
//! # Public API
//! - [`ShellRegistry`] — owns the named shell map.
//! - [`TerminalInfo`] — lightweight snapshot for tool listings.
//! - [`SpawnedBy`] — origin tag for tab-strip styling.
//!
//! # Dependencies
//! `tokio`, [`crate::shell`].

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Result, bail};
use tokio::sync::mpsc;

use crate::shell::{Shell, ShellIo, spawnShell};

/// Default name for the first shell every session starts with.
pub const MAIN_NAME: &str = "main";

/// Who initiated a shell's creation. Drives the tab strip's coloring
/// and lets us track agent-spawned vs user-spawned terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnedBy {
    User,
    Agent,
}

/// Snapshot of a shell entry for the `terminalList` tool and `/term list`.
#[derive(Debug, Clone)]
pub struct TerminalInfo {
    pub name: String,
    pub spawnedBy: SpawnedBy,
    pub ageSecs: u64,
    /// Whether this is the current default target for shell-using tools.
    pub activeForAgent: bool,
}

/// A named shell entry inside the registry.
struct ShellEntry {
    shell: Shell,
    spawnedBy: SpawnedBy,
    spawnedAt: Instant,
}

/// Named map of shells keyed by display name. Owns each [`Shell`] handle;
/// the matching [`ShellIo`] is delivered to the harness via `ioTx` at
/// spawn time.
pub struct ShellRegistry {
    shells: HashMap<String, ShellEntry>,
    /// Insertion order — drives tab strip ordering. `main` always first.
    order: Vec<String>,
    activeForAgent: String,
    nextNameSeq: u32,
    ioTx: mpsc::Sender<(String, ShellIo, SpawnedBy)>,
    /// Default size for newly-spawned shells. The harness sends an
    /// immediate resize once the new ShellIo is connected to its
    /// rendered area, so this is a placeholder until that arrives.
    defaultCols: u16,
    defaultRows: u16,
}

impl Drop for ShellRegistry {
    /// Tear down every live shell when the registry drops. Without
    /// this, orphan PTYs would linger past session shutdown (and
    /// past unit-test runs, hanging the test harness).
    fn drop(&mut self) {
        for entry in self.shells.values() {
            entry.shell.shutdown();
        }
    }
}

impl ShellRegistry {
    /// Create a registry seeded with the `main` shell.
    ///
    /// The caller receives the registry plus the matching ShellIo for
    /// `main` synchronously (so a TUI can wire it up before any other
    /// activity). Subsequent spawns deliver ShellIos through `ioTx`.
    pub fn newWithMain(
        cols: u16,
        rows: u16,
        ioTx: mpsc::Sender<(String, ShellIo, SpawnedBy)>,
    ) -> Result<(Self, ShellIo)> {
        let (shell, io) = spawnShell(cols, rows)?;
        let mut shells = HashMap::new();
        shells.insert(
            MAIN_NAME.into(),
            ShellEntry {
                shell,
                spawnedBy: SpawnedBy::User,
                spawnedAt: Instant::now(),
            },
        );
        let registry = Self {
            shells,
            order: vec![MAIN_NAME.into()],
            activeForAgent: MAIN_NAME.into(),
            nextNameSeq: 2,
            ioTx,
            defaultCols: cols,
            defaultRows: rows,
        };
        Ok((registry, io))
    }

    /// Spawn a new shell. If `name` is None, the registry generates one.
    /// Returns the resolved name; the matching ShellIo is delivered to
    /// the harness on `ioTx`. Fails (and rolls back the spawn) if the
    /// harness channel is full or closed — an invisible PTY is a bug
    /// in the making.
    pub async fn spawn(&mut self, name: Option<String>, spawnedBy: SpawnedBy) -> Result<String> {
        let name = match name {
            Some(n) => {
                if !isValidName(&n) {
                    bail!(
                        "invalid terminal name '{n}' — letters, digits, dashes, underscores only"
                    );
                }
                if self.shells.contains_key(&n) {
                    bail!("terminal '{n}' already exists");
                }
                n
            }
            None => self.generateName(),
        };

        let (shell, io) = spawnShell(self.defaultCols, self.defaultRows)?;

        // Hand the io to the harness BEFORE recording the entry. If the
        // channel is closed/full, abort: the PTY hasn't been tracked
        // anywhere, so dropping `shell` here is the cleanup. The shell's
        // Drop will trigger shutdown when it goes out of scope.
        if let Err(e) = self.ioTx.send((name.clone(), io, spawnedBy)).await {
            shell.shutdown();
            bail!("failed to deliver new terminal io to harness: {e}");
        }

        self.shells.insert(
            name.clone(),
            ShellEntry {
                shell,
                spawnedBy,
                spawnedAt: Instant::now(),
            },
        );
        self.order.push(name.clone());

        Ok(name)
    }

    /// Kill a named shell: send the shutdown signal to its driver task
    /// (which closes the PTY and sends SIGHUP to the process group),
    /// then remove the entry. The harness sees outputRx close and
    /// removes the corresponding tab.
    pub fn kill(&mut self, name: &str) -> Result<()> {
        // Refuse to kill the last live terminal regardless of its name.
        // The earlier guard only fired when the doomed shell was `main`,
        // so a user/agent could `kill main` (leaving e.g. `term2`) then
        // `kill term2` and end up with an empty registry — at which
        // point `activeForAgent` would fall back to the literal `"main"`
        // name even though no shell with that name exists, and the deck
        // would panic looking it up.
        if self.shells.len() == 1 && self.shells.contains_key(name) {
            bail!("cannot kill the last terminal");
        }
        let entry = match self.shells.remove(name) {
            Some(e) => e,
            None => bail!("no terminal named '{name}'"),
        };
        entry.shell.shutdown();
        self.order.retain(|n| n != name);
        // If we just killed the active terminal, fall back to main
        // when present, otherwise to whatever remains.
        if self.activeForAgent == name {
            self.activeForAgent = if self.shells.contains_key(MAIN_NAME) {
                MAIN_NAME.into()
            } else {
                self.order
                    .first()
                    .cloned()
                    .unwrap_or_else(|| MAIN_NAME.into())
            };
        }
        Ok(())
    }

    /// Look up a shell by name, falling back to the active default when
    /// `name` is None. Returns a clone — `Shell` is cheap to clone
    /// (just channels and `Arc`s).
    pub fn shellFor(&self, name: Option<&str>) -> Option<Shell> {
        let key = name.unwrap_or(self.activeForAgent.as_str());
        self.shells.get(key).map(|e| e.shell.clone())
    }

    /// Set the default target for shell-using tool calls.
    pub fn setActiveForAgent(&mut self, name: &str) -> Result<()> {
        if !self.shells.contains_key(name) {
            bail!("no terminal named '{name}'");
        }
        self.activeForAgent = name.into();
        Ok(())
    }

    /// Name of the current default target.
    pub fn activeForAgent(&self) -> &str {
        &self.activeForAgent
    }

    /// Snapshot of every shell entry in insertion order (`main` first,
    /// then user/agent spawns in the order they were created).
    pub fn list(&self) -> Vec<TerminalInfo> {
        self.order
            .iter()
            .filter_map(|name| {
                let e = self.shells.get(name)?;
                Some(TerminalInfo {
                    name: name.clone(),
                    spawnedBy: e.spawnedBy,
                    ageSecs: e.spawnedAt.elapsed().as_secs(),
                    activeForAgent: name == &self.activeForAgent,
                })
            })
            .collect()
    }

    /// All terminal names in display order.
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// Number of live shells.
    pub fn len(&self) -> usize {
        self.shells.len()
    }

    /// True when the named shell exists.
    pub fn contains(&self, name: &str) -> bool {
        self.shells.contains_key(name)
    }

    /// Generate a unique auto name like `term2`, `term3`, ... Skips any
    /// names already in use.
    fn generateName(&mut self) -> String {
        loop {
            let candidate = format!("term{}", self.nextNameSeq);
            self.nextNameSeq += 1;
            if !self.shells.contains_key(&candidate) {
                return candidate;
            }
        }
    }
}

/// Names allowed: letters, digits, `-`, `_`. Must be 1-32 chars.
fn isValidName(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 32
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    type IoMsg = (String, ShellIo, SpawnedBy);

    /// Build a registry + a drainer that absorbs every new ShellIo to
    /// keep the spawn channel from filling. Returns the registry and
    /// the main shell's io.
    fn fixture() -> (ShellRegistry, ShellIo) {
        let (tx, mut rx) = mpsc::channel::<IoMsg>(16);
        let (reg, mainIo) = ShellRegistry::newWithMain(80, 24, tx).expect("newWithMain");
        // Drain spawned io tuples in the background so registry sends
        // don't block. Drop the io immediately — the shell's driver
        // task notices its outputTx receiver is gone and unwinds.
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        (reg, mainIo)
    }

    #[tokio::test]
    async fn mainExistsAfterConstruction() {
        let (reg, _mainIo) = fixture();
        assert_eq!(reg.len(), 1);
        assert!(reg.contains(MAIN_NAME));
        assert_eq!(reg.activeForAgent(), MAIN_NAME);
    }

    #[tokio::test]
    async fn spawnAutoName() {
        let (mut reg, _mainIo) = fixture();
        let name = reg.spawn(None, SpawnedBy::Agent).await.expect("spawn");
        assert_eq!(name, "term2");
        assert_eq!(reg.len(), 2);
    }

    #[tokio::test]
    async fn spawnNamed() {
        let (mut reg, _mainIo) = fixture();
        let name = reg
            .spawn(Some("build".into()), SpawnedBy::Agent)
            .await
            .expect("spawn");
        assert_eq!(name, "build");
        assert!(reg.contains("build"));
    }

    #[tokio::test]
    async fn spawnDuplicateRejected() {
        let (mut reg, _mainIo) = fixture();
        reg.spawn(Some("build".into()), SpawnedBy::Agent)
            .await
            .unwrap();
        assert!(
            reg.spawn(Some("build".into()), SpawnedBy::Agent)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn invalidNameRejected() {
        let (mut reg, _mainIo) = fixture();
        assert!(
            reg.spawn(Some("with space".into()), SpawnedBy::Agent)
                .await
                .is_err()
        );
        assert!(reg.spawn(Some("".into()), SpawnedBy::Agent).await.is_err());
    }

    #[tokio::test]
    async fn killRemovesShell() {
        let (mut reg, _mainIo) = fixture();
        reg.spawn(Some("build".into()), SpawnedBy::Agent)
            .await
            .unwrap();
        reg.kill("build").unwrap();
        assert!(!reg.contains("build"));
        assert_eq!(reg.len(), 1);
    }

    #[tokio::test]
    async fn cannotKillLastShell() {
        let (mut reg, _mainIo) = fixture();
        assert!(reg.kill(MAIN_NAME).is_err());
    }

    #[tokio::test]
    async fn cannotKillLastShellWhenItIsNotMain() {
        // Scenario that earlier escaped the guard: spawn `term2`, then
        // kill `main` (registry now has just `term2`), then try to kill
        // `term2`. The old guard only fired for `main`, so this used to
        // succeed and leave the registry empty — at which point
        // `activeForAgent` would fall back to the literal "main" name
        // and the deck would panic looking it up.
        let (mut reg, _mainIo) = fixture();
        reg.spawn(Some("term2".into()), SpawnedBy::Agent)
            .await
            .unwrap();
        reg.kill(MAIN_NAME)
            .expect("kill main when term2 still alive");
        assert_eq!(reg.len(), 1);
        assert!(reg.kill("term2").is_err(), "must refuse to kill last shell");
        assert_eq!(reg.len(), 1);
        assert!(reg.contains("term2"));
    }

    #[tokio::test]
    async fn killActiveFallsBack() {
        let (mut reg, _mainIo) = fixture();
        reg.spawn(Some("build".into()), SpawnedBy::Agent)
            .await
            .unwrap();
        reg.setActiveForAgent("build").unwrap();
        reg.kill("build").unwrap();
        assert_eq!(reg.activeForAgent(), MAIN_NAME);
    }

    #[tokio::test]
    async fn shellForResolvesActive() {
        let (mut reg, _mainIo) = fixture();
        reg.spawn(Some("build".into()), SpawnedBy::Agent)
            .await
            .unwrap();
        reg.setActiveForAgent("build").unwrap();
        assert!(reg.shellFor(None).is_some());
        assert!(reg.shellFor(Some("main")).is_some());
        assert!(reg.shellFor(Some("nope")).is_none());
    }

    #[tokio::test]
    async fn listIsInsertionOrdered() {
        let (mut reg, _mainIo) = fixture();
        reg.spawn(Some("build".into()), SpawnedBy::Agent)
            .await
            .unwrap();
        reg.spawn(Some("logs".into()), SpawnedBy::User)
            .await
            .unwrap();
        let names: Vec<String> = reg.list().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["main", "build", "logs"]);
    }
}
