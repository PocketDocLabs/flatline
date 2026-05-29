#![allow(non_snake_case)]

use ratatui::style::Color;

pub fn terminalRunImpactGlyphColor(
    impact: construct::storage::TerminalRunImpact,
) -> (&'static str, Color) {
    match impact {
        construct::storage::TerminalRunImpact::Delete => ("\u{2620}\u{FE0E}", Color::Red),
        construct::storage::TerminalRunImpact::MajorMod => {
            ("\u{26A0}\u{FE0E}", Color::Rgb(200, 140, 40))
        }
        construct::storage::TerminalRunImpact::MinorMod => {
            ("\u{2691}\u{FE0E}", Color::Rgb(180, 160, 80))
        }
        construct::storage::TerminalRunImpact::Read => ("\u{2315}", Color::Rgb(80, 160, 200)),
    }
}

pub fn shellImpactGlyphColorForTool(impact: construct::tool::ShellImpact) -> (&'static str, Color) {
    terminalRunImpactGlyphColor(construct::storage::TerminalRunImpact::from(&impact))
}
