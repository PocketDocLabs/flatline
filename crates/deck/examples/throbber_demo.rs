//! Standalone throbber demo — renders the blob animation in the terminal.
//!
//! Run: cargo run -p deck --example throbber_demo
//!
//! Pass a shape index (0-3) to skip straight to that shape:
//!   cargo run -p deck --example throbber_demo -- 1

use std::io::{self, Write};
use std::time::{Duration, Instant};

// Pull in the throbber module from the deck crate's lib... except deck
// is a binary crate. We'll just inline the throbber module path.
#[path = "../src/throbber.rs"]
mod throbber;
use throbber::Throbber;

fn main() {
    let forceShape: Option<usize> = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok());

    let mut throbber = Throbber::new();

    // If a shape index was given, set ticksUntilShape to 0 so it triggers immediately.
    if forceShape.is_some() {
        throbber.forceNextShape(forceShape);
    }

    let tick = Duration::from_millis(125);
    let mut stdout = io::stdout();

    // Hide cursor.
    print!("\x1b[?25l");

    loop {
        let start = Instant::now();

        let lines = throbber.renderLines();
        // Move to top-left, clear, render.
        print!("\x1b[H\x1b[2J");
        for line in &lines {
            let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            println!("{text}");
        }
        let _ = stdout.flush();

        throbber.tick();

        let elapsed = start.elapsed();
        if elapsed < tick {
            std::thread::sleep(tick - elapsed);
        }
    }
}
