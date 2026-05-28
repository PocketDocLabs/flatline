#![allow(non_snake_case)]

//! Layout discovery and persistence.
//!
//! Walks the cwd parent chain looking for `.flatline/layout.toml`,
//! mirroring how `AGENTS.md` is discovered. Falls back to
//! `~/.config/flatline/layout.toml`. `writeLayout` serializes a
//! `Layout` back to disk as TOML so the Ctrl+O panel can persist
//! user-chosen presets.
//!
//! Files that parse as `Layout` but fail [`Layout::isCanonicalPhase1`]
//! are reported as `Err` so the caller can fall back to the default
//! rather than render an empty pane.
//!
//! # Public API
//! - [`discoverLayout`] — return the resolved path + parse result, or `None`
//! - [`writeLayout`] — serialize a layout to a TOML file
//! - [`DiscoveredLayout`] — discovery result
//!
//! On a malformed file the discovery returns `Some(DiscoveredLayout { result: Err(_) })`
//! so the caller can fall back to [`Layout::defaultPhase1`] and surface a
//! one-line notice with the parse error.

use std::path::{Path, PathBuf};

use anyhow::Result;

use super::Layout;

/// File name we look for during walk-up discovery.
pub const LAYOUT_FILENAME: &str = "layout.toml";
/// Per-project marker directory, mirrors `.flatline` used elsewhere.
pub const PROJECT_DIR: &str = ".flatline";
/// User-config fallback path under `~/.config/flatline/layout.toml`.
pub const CONFIG_SUBDIR: &str = "flatline";

/// Result of a discovery walk. `result` is `Err` when a file was
/// located but failed to parse — the caller falls back to a default
/// and surfaces the message.
#[derive(Debug)]
pub struct DiscoveredLayout {
    pub path: PathBuf,
    pub result: Result<Layout, String>,
}

/// Walk up from `cwd` checking each ancestor for `.flatline/layout.toml`.
/// Falls back to `~/.config/flatline/layout.toml`. Returns `None` if
/// no candidate file exists.
pub fn discoverLayout(cwd: &Path) -> Option<DiscoveredLayout> {
    if let Some(found) = walkUp(cwd) {
        return Some(loadAt(found));
    }
    if let Some(fallback) = configFallbackPath()
        && fallback.exists()
    {
        return Some(loadAt(fallback));
    }
    None
}

/// Walk up from `start` looking for `<dir>/.flatline/layout.toml`.
/// Returns the first hit (closest to cwd wins).
fn walkUp(start: &Path) -> Option<PathBuf> {
    let mut dir = start;
    loop {
        let candidate = dir.join(PROJECT_DIR).join(LAYOUT_FILENAME);
        if candidate.exists() {
            return Some(candidate);
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return None,
        }
    }
}

/// `~/.config/flatline/layout.toml` (XDG-compliant on Linux; macOS uses
/// the same path via `dirs::config_dir`).
pub fn configFallbackPath() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join(CONFIG_SUBDIR).join(LAYOUT_FILENAME))
}

fn loadAt(path: PathBuf) -> DiscoveredLayout {
    let result = std::fs::read_to_string(&path)
        .map_err(|e| format!("read failed: {e}"))
        .and_then(|s| toml::from_str::<Layout>(&s).map_err(|e| format!("parse failed: {e}")))
        .and_then(|layout| {
            if layout.isCanonicalPhase1() {
                Ok(layout)
            } else {
                Err("unsupported shape: expected horizontal Split of \
                     terminal Tabs and AgentPanel"
                    .into())
            }
        });
    DiscoveredLayout { path, result }
}

/// Serialize `layout` to `path` as TOML. Creates the parent directory
/// if it doesn't already exist.
pub fn writeLayout(path: &Path, layout: &Layout) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(layout)?;
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{Orient, WindowId};
    use tempfile::TempDir;

    #[test]
    fn roundtripDefaultPhase1() {
        // Hand-write the default through serde and read it back. The
        // result must be structurally identical (we don't `derive(Eq)`
        // because of `f32` ratios, so compare via re-rendered areas).
        let original = Layout::defaultPhase1();
        let text = toml::to_string_pretty(&original).expect("serialize");
        let parsed: Layout = toml::from_str(&text).expect("parse");

        let area = ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 30,
        };
        let a = original.computeAreas(area);
        let b = parsed.computeAreas(area);
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b.iter()) {
            assert_eq!(ai.rect, bi.rect);
            assert_eq!(ai.window, bi.window);
        }
    }

    #[test]
    fn discoveryWalksUpToParent() {
        // Layout file in `<root>/.flatline/layout.toml`. Discovery
        // from `<root>/sub/sub2` should find it.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let layoutDir = root.join(PROJECT_DIR);
        std::fs::create_dir(&layoutDir).unwrap();
        let layoutPath = layoutDir.join(LAYOUT_FILENAME);
        writeLayout(&layoutPath, &Layout::defaultPhase1()).unwrap();

        let deep = root.join("sub").join("sub2");
        std::fs::create_dir_all(&deep).unwrap();

        let found = discoverLayout(&deep).expect("layout should be discovered");
        // The resolved path may go through symlinks (TempDir on macOS
        // uses /var which links to /private/var); compare canonicalized
        // forms so the test holds across platforms.
        assert_eq!(
            std::fs::canonicalize(&found.path).unwrap(),
            std::fs::canonicalize(&layoutPath).unwrap(),
        );
        assert!(
            found.result.is_ok(),
            "well-formed file should parse: {found:?}"
        );
    }

    #[test]
    fn discoveryReturnsNoneWhenNothingExists() {
        let tmp = TempDir::new().unwrap();
        // No layout file anywhere in the chain — and the fallback
        // path is highly unlikely to exist in CI. If it does, this
        // test reports it; either way the test logic stays valid.
        let found = discoverLayout(tmp.path());
        let fallbackExists = configFallbackPath().map(|p| p.exists()).unwrap_or(false);
        if fallbackExists {
            // Walk-up missed, fallback existed — discovery should return Some.
            assert!(found.is_some());
        } else {
            assert!(found.is_none(), "no layout anywhere should produce None");
        }
    }

    #[test]
    fn discoveryReturnsParseErrorForMalformedFile() {
        let tmp = TempDir::new().unwrap();
        let layoutDir = tmp.path().join(PROJECT_DIR);
        std::fs::create_dir(&layoutDir).unwrap();
        let layoutPath = layoutDir.join(LAYOUT_FILENAME);
        // Write garbage that the toml parser will choke on.
        std::fs::write(&layoutPath, b"= not = a = valid = toml = file =").unwrap();

        let found = discoverLayout(tmp.path()).expect("file should be discovered");
        match found.result {
            Err(msg) => assert!(
                msg.contains("parse failed"),
                "expected parse failure, got: {msg}",
            ),
            Ok(_) => panic!("malformed file should not parse"),
        }
    }

    #[test]
    fn discoveryRejectsUnsupportedShape() {
        // A vertical split is well-formed TOML and deserializes fine
        // but app.rs can't render it. Discovery must surface that as
        // an error so the caller falls back to default instead of
        // showing an invisible pane.
        let tmp = TempDir::new().unwrap();
        let layoutDir = tmp.path().join(PROJECT_DIR);
        std::fs::create_dir(&layoutDir).unwrap();
        let layoutPath = layoutDir.join(LAYOUT_FILENAME);

        let bogus = Layout::Split {
            orient: Orient::Vertical,
            ratio: 0.5,
            a: Box::new(Layout::Window(WindowId::Terminal("main".into()))),
            b: Box::new(Layout::Window(WindowId::AgentPanel)),
        };
        let text = toml::to_string_pretty(&bogus).unwrap();
        std::fs::write(&layoutPath, text).unwrap();

        let found = discoverLayout(tmp.path()).expect("file should be discovered");
        match found.result {
            Err(msg) => assert!(
                msg.contains("unsupported shape"),
                "expected unsupported-shape error, got: {msg}",
            ),
            Ok(_) => panic!("unsupported-shape layout should be rejected"),
        }
    }

    #[test]
    fn writeLayoutCreatesParentDirectory() {
        let tmp = TempDir::new().unwrap();
        // Path inside a directory that doesn't exist yet.
        let path = tmp.path().join("nested").join("layout.toml");
        writeLayout(&path, &Layout::defaultPhase1()).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn closestLayoutWinsWalkUp() {
        // A layout closer to cwd should override one further away.
        let tmp = TempDir::new().unwrap();
        let outer = tmp.path();
        let inner = outer.join("project");
        std::fs::create_dir(&inner).unwrap();

        // Outer layout: ratio 0.6 (the default)
        writeLayout(
            &outer.join(PROJECT_DIR).join(LAYOUT_FILENAME),
            &Layout::defaultPhase1(),
        )
        .unwrap();

        // Inner layout: a different ratio so we can tell them apart.
        let innerLayout = Layout::Split {
            orient: Orient::Horizontal,
            ratio: 0.75,
            a: Box::new(Layout::Tabs {
                active: 0,
                children: vec![Layout::Window(WindowId::Terminal("main".into()))],
            }),
            b: Box::new(Layout::Window(WindowId::AgentPanel)),
        };
        writeLayout(&inner.join(PROJECT_DIR).join(LAYOUT_FILENAME), &innerLayout).unwrap();

        let found = discoverLayout(&inner).expect("layout discovered");
        let layout = found.result.expect("parses");
        // Render at 100 wide: 0.75 → terminal width 75, agent width 25.
        let area = ratatui::layout::Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 30,
        };
        let areas = layout.computeAreas(area);
        let termRect = areas
            .iter()
            .find(|a| matches!(a.window, WindowId::Terminal(_)))
            .unwrap()
            .rect;
        assert_eq!(
            termRect.width, 75,
            "closer (.flatline/) should win, got {termRect:?}"
        );
    }
}
