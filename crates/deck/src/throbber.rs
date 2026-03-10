//! Living braille blob throbber animation.
//!
//! Renders a pulsing organic blob on a 4x2 braille character canvas
//! (8 columns x 8 rows of dots). Uses a distance-from-center algorithm
//! with time-varying threshold and noise for an alive, breathing feel.
//!
//! # Public API
//! - [`Throbber`] — animation state and rendering
//!
//! # Dependencies
//! `ratatui`

use ratatui::text::{Line, Span};
use ratatui::style::{Color, Style};

/// Braille blob animation state.
pub struct Throbber {
    /// Current frame counter (wraps around).
    frame: u32,
    /// Noise seed for organic variation.
    seed: u32,
}

// Canvas dimensions in dots.
const COLS: usize = 8;
const ROWS: usize = 8;
const CENTER_X: f32 = 3.5;
const CENTER_Y: f32 = 3.5;

impl Throbber {
    pub fn new() -> Self {
        Self { frame: 0, seed: 0xDEAD }
    }

    /// Advance one animation tick. Call at ~8 FPS.
    pub fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        // Slowly evolve the noise seed for variation.
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
    }

    /// Render the blob as two lines of 4 braille characters each.
    pub fn renderLines(&self) -> [Line<'static>; 2] {
        let grid = self.computeGrid();

        let style = Style::default().fg(Color::Magenta);
        let mut topChars = String::with_capacity(4);
        let mut bottomChars = String::with_capacity(4);

        // Top row: braille chars covering grid rows 0-3.
        for col in 0..4 {
            let ch = gridToBraille(&grid, col * 2, 0);
            topChars.push(ch);
        }
        // Bottom row: braille chars covering grid rows 4-7.
        for col in 0..4 {
            let ch = gridToBraille(&grid, col * 2, 4);
            bottomChars.push(ch);
        }

        [
            Line::from(Span::styled(topChars, style)),
            Line::from(Span::styled(bottomChars, style)),
        ]
    }

    /// Compute the 8x8 dot grid for the current frame.
    fn computeGrid(&self) -> [[bool; COLS]; ROWS] {
        let mut grid = [[false; COLS]; ROWS];
        let t = self.frame as f32 * 0.15;

        // Pulsing radius.
        let baseRadius = 2.2;
        let pulseAmplitude = 1.5;
        let radius = baseRadius + pulseAmplitude * t.sin();

        // Secondary pulse for asymmetry.
        let radius2 = 0.4 * (t * 1.7 + 0.8).sin();

        for row in 0..ROWS {
            for col in 0..COLS {
                let dx = col as f32 - CENTER_X;
                let dy = row as f32 - CENTER_Y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Angle-dependent distortion for organic shape.
                let angle = dy.atan2(dx);
                let wobble = 0.3 * (angle * 3.0 + t * 0.9).sin()
                    + 0.2 * (angle * 5.0 - t * 1.3).sin();

                // Per-cell noise.
                let cellNoise = self.noise(col as u32, row as u32) * 0.5;

                let threshold = radius + radius2 + wobble + cellNoise;
                grid[row][col] = dist < threshold;
            }
        }
        grid
    }

    /// Simple deterministic noise based on cell position and frame.
    fn noise(&self, x: u32, y: u32) -> f32 {
        let h = x
            .wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(self.seed);
        let h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        let h = h ^ (h >> 16);
        // Map to [-1, 1].
        (h as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Encode a 2x4 region of the grid starting at (gridCol, gridRow) as a braille character.
///
/// Braille dot positions within a character:
/// ```text
///   col0  col1
///   0x01  0x08   row+0
///   0x02  0x10   row+1
///   0x04  0x20   row+2
///   0x40  0x80   row+3
/// ```
fn gridToBraille(grid: &[[bool; COLS]; ROWS], gridCol: usize, gridRow: usize) -> char {
    let mut bits: u8 = 0;

    let dotMap: [(usize, usize, u8); 8] = [
        (0, 0, 0x01),
        (0, 1, 0x02),
        (0, 2, 0x04),
        (0, 3, 0x40),
        (1, 0, 0x08),
        (1, 1, 0x10),
        (1, 2, 0x20),
        (1, 3, 0x80),
    ];

    for &(dc, dr, bit) in &dotMap {
        let c = gridCol + dc;
        let r = gridRow + dr;
        if c < COLS && r < ROWS && grid[r][c] {
            bits |= bit;
        }
    }

    char::from_u32(0x2800 + bits as u32).unwrap_or('\u{2800}')
}
