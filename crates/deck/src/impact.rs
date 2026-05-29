#![allow(non_snake_case)]

use ratatui::style::Color;

pub fn shellImpactGlyphColor(impact: &str) -> (&'static str, Color) {
    match impact {
        "delete" => ("\u{2620}\u{FE0E}", Color::Red),
        "majorMod" => ("\u{26A0}\u{FE0E}", Color::Rgb(200, 140, 40)),
        "minorMod" => ("\u{2691}\u{FE0E}", Color::Rgb(180, 160, 80)),
        _ => ("\u{2315}", Color::Rgb(80, 160, 200)),
    }
}

pub fn shellImpactGlyphColorForTool(impact: construct::tool::ShellImpact) -> (&'static str, Color) {
    match impact {
        construct::tool::ShellImpact::Delete => shellImpactGlyphColor("delete"),
        construct::tool::ShellImpact::MajorMod => shellImpactGlyphColor("majorMod"),
        construct::tool::ShellImpact::MinorMod => shellImpactGlyphColor("minorMod"),
        construct::tool::ShellImpact::Read => shellImpactGlyphColor("read"),
    }
}
