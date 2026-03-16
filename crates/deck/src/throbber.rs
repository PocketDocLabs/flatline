//! Living braille blob throbber with easter-egg shape morphing.
//!
//! The blob is a center-biased Game of Life simulation on a 4x2 braille
//! character canvas (8x8 dots). Cells near the center are nudged toward
//! life, cells at the edges toward death, creating a pulsing organic blob.
//!
//! Occasionally the blob crystallizes into a recognizable shape (heart,
//! skull, lightning, star) via a radial sweep. On dissolve, the shape
//! pixels are injected into the GoL grid and the simulation naturally
//! pulls them back into blob form.
//!
//! # Public API
//! - [`Throbber`] — animation state and rendering
//!
//! # Dependencies
//! `ratatui`

use ratatui::text::{Line, Span};
use ratatui::style::{Color, Style};

// Canvas dimensions in dots.
const COLS: usize = 8;
const ROWS: usize = 8;
const CENTER_X: f32 = 3.5;
const CENTER_Y: f32 = 3.5;

type Grid = [[bool; COLS]; ROWS];

// Phase durations in ticks (~8 FPS).
const FORM_TICKS: u32 = 8;
const HOLD_TICKS: u32 = 4;
const DISSOLVE_TICKS: u32 = 8;

// ── Shape templates ──────────────────────────────────────────────
// Each row is a bitmask, MSB = column 0 (leftmost).

fn bitsToGrid(bits: [u8; 8]) -> Grid {
    let mut grid = [[false; COLS]; ROWS];
    for row in 0..ROWS {
        for col in 0..COLS {
            grid[row][col] = bits[row] & (0x80 >> col) != 0;
        }
    }
    grid
}

const SHAPE_COUNT: usize = 4;
const EYE_IDX: usize = 1;

// Eye hold is longer to fit: open → look around → blink → open.
// 16 ticks = 2 seconds at 8fps.
const EYE_HOLD_TICKS: u32 = 16;

/// Animated eye frames for the hold phase.
fn eyeFrame(t: u32) -> Grid {
    match t {
        // Ticks 0-3: open (center pupil).
        0..=3 => shape(EYE_IDX),
        // Ticks 4-5: look left (pupil shifts left).
        4..=5 => bitsToGrid([
            0b00000000,
            0b00011000,
            0b01111110,
            0b11011111,
            0b11001111,
            0b01111110,
            0b00011000,
            0b00000000,
        ]),
        // Ticks 6-7: look right (pupil shifts right).
        6..=7 => bitsToGrid([
            0b00000000,
            0b00011000,
            0b01111110,
            0b11111011,
            0b11110011,
            0b01111110,
            0b00011000,
            0b00000000,
        ]),
        // Ticks 8-9: center again.
        8..=9 => shape(EYE_IDX),
        // Tick 10: half-closed.
        10 => bitsToGrid([
            0b00000000,
            0b00000000,
            0b00111100,
            0b01111110,
            0b01111110,
            0b00111100,
            0b00000000,
            0b00000000,
        ]),
        // Tick 11: closed.
        11 => bitsToGrid([
            0b00000000,
            0b00000000,
            0b00000000,
            0b01111110,
            0b01111110,
            0b00000000,
            0b00000000,
            0b00000000,
        ]),
        // Tick 12: half-open.
        12 => bitsToGrid([
            0b00000000,
            0b00000000,
            0b00111100,
            0b01111110,
            0b01111110,
            0b00111100,
            0b00000000,
            0b00000000,
        ]),
        // Ticks 13+: open again.
        _ => shape(EYE_IDX),
    }
}

fn shape(idx: usize) -> Grid {
    match idx {
        // Heart.
        0 => bitsToGrid([
            0b01100110,
            0b11111111,
            0b11111111,
            0b11111111,
            0b01111110,
            0b00111100,
            0b00011000,
            0b00000000,
        ]),
        // Eye (almond with asymmetric pupil).
        1 => bitsToGrid([
            0b00000000,
            0b00011000,
            0b01111110,
            0b11101111,
            0b11100111,
            0b01111110,
            0b00011000,
            0b00000000,
        ]),
        // Lightning bolt.
        2 => bitsToGrid([
            0b00001100,
            0b00011000,
            0b00110000,
            0b01111110,
            0b00001100,
            0b00011000,
            0b00110000,
            0b01100000,
        ]),
        // Heartbeat (EKG waveform, vertical trace).
        _ => bitsToGrid([
            0b00010000,
            0b00010000,
            0b00010100,
            0b01010100,
            0b10101011,
            0b00001000,
            0b00001000,
            0b00001000,
        ]),
    }
}

// ── Phase state machine ──────────────────────────────────────────

#[derive(Clone, Copy)]
enum Phase {
    /// Normal blob — GoL running with center bias.
    Blob,
    /// Radial sweep: blob crystallizing into shape.
    Forming(u32),
    /// Shape fully formed, holding.
    Holding(u32),
    /// GoL rules transitioning from shape-friendly to normal.
    Dissolving(u32),
}

// ── Throbber ─────────────────────────────────────────────────────

/// Braille blob animation state — center-biased Game of Life.
pub struct Throbber {
    /// Noise seed for organic variation (evolves each tick).
    seed: u32,
    /// Current animation phase.
    phase: Phase,
    /// Which shape template is active during a morph event.
    shapeIdx: usize,
    /// Ticks remaining until next shape event (blob mode only).
    ticksUntilShape: u32,
    /// Override for the next shape index (demo/testing).
    forcedShape: Option<usize>,
    /// The live GoL grid — this IS the blob.
    grid: Grid,
}

/// Generate an initial blob-shaped grid seeded from center.
fn initGrid() -> Grid {
    let mut grid = [[false; COLS]; ROWS];
    for row in 0..ROWS {
        for col in 0..COLS {
            let dx = col as f32 - CENTER_X;
            let dy = row as f32 - CENTER_Y;
            let dist = (dx * dx + dy * dy).sqrt();
            grid[row][col] = dist < 2.8;
        }
    }
    grid
}

impl Throbber {
    pub fn new() -> Self {
        Self {
            seed: 0xDEAD,
            phase: Phase::Blob,
            shapeIdx: 0,
            // First shape after 20-35 seconds of throbber activity.
            ticksUntilShape: 200,
            forcedShape: None,
            grid: initGrid(),
        }
    }

    /// Force the next shape event. If `idx` is Some, use that shape;
    /// otherwise pick randomly. Sets countdown to 0 so it triggers next tick.
    /// Force the next shape event. If `idx` is Some, use that shape;
    /// otherwise pick randomly. Sets countdown to 0 so it triggers next tick.
    pub fn forceNextShape(&mut self, idx: Option<usize>) {
        self.ticksUntilShape = 0;
        self.forcedShape = idx.map(|i| i % SHAPE_COUNT);
    }

    /// Advance one animation tick. Call at ~8 FPS.
    pub fn tick(&mut self) {
        // Evolve the noise seed.
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);

        // Always step the GoL simulation.
        // During dissolve, lerp rules from permissive (shape-friendly) to normal.
        let leniency = match self.phase {
            Phase::Dissolving(t) => 1.0 - (t as f32 / DISSOLVE_TICKS as f32),
            _ => 0.0,
        };
        self.grid = self.stepBlobGol(leniency);

        // Advance the phase state machine.
        self.phase = match self.phase {
            Phase::Blob => {
                if self.ticksUntilShape == 0 {
                    self.shapeIdx = self.forcedShape
                        .take()
                        .unwrap_or_else(|| (self.seed as usize / 7) % SHAPE_COUNT);
                    Phase::Forming(0)
                } else {
                    self.ticksUntilShape -= 1;
                    Phase::Blob
                }
            }
            Phase::Forming(t) => {
                if t >= FORM_TICKS {
                    Phase::Holding(0)
                } else {
                    Phase::Forming(t + 1)
                }
            }
            Phase::Holding(t) => {
                let maxHold = if self.shapeIdx == EYE_IDX { EYE_HOLD_TICKS } else { HOLD_TICKS };
                if t >= maxHold {
                    // Inject the shape into the GoL grid.
                    self.grid = shape(self.shapeIdx);
                    Phase::Dissolving(0)
                } else {
                    Phase::Holding(t + 1)
                }
            }
            Phase::Dissolving(t) => {
                if t >= DISSOLVE_TICKS {
                    self.ticksUntilShape = 180 + (self.seed % 121);
                    Phase::Blob
                } else {
                    Phase::Dissolving(t + 1)
                }
            }
        };
    }

    /// Render the blob as two lines of 4 braille characters each.
    pub fn renderLines(&self) -> [Line<'static>; 2] {
        let grid = self.displayGrid();

        let style = Style::default().fg(Color::Magenta);
        let mut topChars = String::with_capacity(4);
        let mut bottomChars = String::with_capacity(4);

        for col in 0..4 {
            let ch = gridToBraille(&grid, col * 2, 0);
            topChars.push(ch);
        }
        for col in 0..4 {
            let ch = gridToBraille(&grid, col * 2, 4);
            bottomChars.push(ch);
        }

        [
            Line::from(Span::styled(topChars, style)),
            Line::from(Span::styled(bottomChars, style)),
        ]
    }

    /// Grid to display — live GoL grid, or blended during forming/holding.
    fn displayGrid(&self) -> Grid {
        match self.phase {
            Phase::Blob => self.grid,
            Phase::Forming(t) => {
                let progress = t as f32 / FORM_TICKS as f32;
                self.blendGrids(&self.grid, &shape(self.shapeIdx), progress)
            }
            Phase::Holding(t) => {
                if self.shapeIdx == EYE_IDX {
                    eyeFrame(t)
                } else {
                    // Subtle breathing while holding the shape.
                    let breath = 0.92 + 0.08 * (t as f32 * 0.4).sin();
                    self.blendGrids(&self.grid, &shape(self.shapeIdx), breath)
                }
            }
            Phase::Dissolving(_) => self.grid,
        }
    }

    /// One GoL generation with center-biased life injection and edge culling.
    ///
    /// Leniency 0.0 = normal blob rules (aggressive center bias, tight edges).
    /// Leniency 1.0 = shape-friendly rules (wide injection, no edge culling).
    /// During dissolve, leniency lerps 1.0 → 0.0 over DISSOLVE_TICKS so the
    /// shape erodes gradually instead of exploding.
    fn stepBlobGol(&self, leniency: f32) -> Grid {
        let mut next = stepGol(&self.grid);

        // Lerp thresholds between normal (len=0) and permissive (len=1).
        let coreRadius = 1.5 + leniency * 1.0;        // 1.5 → 2.5
        let coreThreshold = 0.0 - leniency * 0.3;     // 0.0 → -0.3
        let midRadius = 2.5 + leniency * 0.5;          // 2.5 → 3.0
        let midThreshold = 0.6 - leniency * 0.1;       // 0.6 → 0.5
        let edgeStart = 3.5 + leniency * 0.5;          // 3.5 → 4.0
        let edgeThreshold = -0.3 + leniency * 0.3;     // -0.3 → 0.0
        let killDist = 4.5 + leniency * 0.5;           // 4.5 → 5.0

        for row in 0..ROWS {
            for col in 0..COLS {
                let dx = col as f32 - CENTER_X;
                let dy = row as f32 - CENTER_Y;
                let dist = (dx * dx + dy * dy).sqrt();
                let noise = self.noise(col as u32, row as u32);

                // Core: spontaneous generation keeps the center alive.
                if dist < coreRadius && noise > coreThreshold {
                    next[row][col] = true;
                }
                // Mid-ring: occasional spontaneous generation.
                if dist < midRadius && !next[row][col] && noise > midThreshold {
                    next[row][col] = true;
                }
                // Edges: cull to prevent infinite growth.
                if dist > edgeStart && noise > edgeThreshold {
                    next[row][col] = false;
                }
                // Hard kill past outer boundary.
                if dist > killDist {
                    next[row][col] = false;
                }
            }
        }
        next
    }

    /// Blend two grids with a radial sweep effect.
    /// Progress 0.0 = pure source, 1.0 = pure target.
    /// Cells near the center morph first.
    fn blendGrids(&self, source: &Grid, target: &Grid, progress: f32) -> Grid {
        let mut result = [[false; COLS]; ROWS];
        let maxDist: f32 = 5.0;

        for row in 0..ROWS {
            for col in 0..COLS {
                let dx = col as f32 - CENTER_X;
                let dy = row as f32 - CENTER_Y;
                let dist = (dx * dx + dy * dy).sqrt();

                let waveFront = progress * (maxDist + 1.5);
                let cellBlend = ((waveFront - dist) / 1.5).clamp(0.0, 1.0);

                let noise = self.noise(col as u32, row as u32) * 0.25;
                let cellBlend = (cellBlend + noise).clamp(0.0, 1.0);

                result[row][col] = if cellBlend > 0.5 {
                    target[row][col]
                } else {
                    source[row][col]
                };
            }
        }
        result
    }

    /// Deterministic noise based on cell position and current seed.
    fn noise(&self, x: u32, y: u32) -> f32 {
        let h = x
            .wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(self.seed);
        let h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        let h = h ^ (h >> 16);
        (h as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Advance a Game of Life grid by one generation (no wrapping).
fn stepGol(grid: &Grid) -> Grid {
    let mut next = [[false; COLS]; ROWS];
    for row in 0..ROWS {
        for col in 0..COLS {
            let n = golNeighbors(grid, row, col);
            next[row][col] = matches!((grid[row][col], n), (true, 2) | (true, 3) | (false, 3));
        }
    }
    next
}

/// Count live neighbors for a cell (8-connected, no wrapping).
fn golNeighbors(grid: &Grid, row: usize, col: usize) -> u8 {
    let mut count = 0u8;
    for dr in [-1i32, 0, 1] {
        for dc in [-1i32, 0, 1] {
            if dr == 0 && dc == 0 { continue; }
            let r = row as i32 + dr;
            let c = col as i32 + dc;
            if r >= 0 && r < ROWS as i32 && c >= 0 && c < COLS as i32
                && grid[r as usize][c as usize]
            {
                count += 1;
            }
        }
    }
    count
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
fn gridToBraille(grid: &Grid, gridCol: usize, gridRow: usize) -> char {
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
