//! Mermaid flowchart parser and Unicode diagram renderer.
//!
//! Parses the `graph`/`flowchart` subset of Mermaid syntax into an
//! intermediate representation, then renders via ascii-dag's Sugiyama
//! layout into styled ratatui spans for display in a scrollable code block.
//!
//! Non-flowchart mermaid types and parse failures return `None` so the
//! caller can fall back to a plain code block.
//!
//! # Public API
//! - [`tryRenderMermaid`] — parse + render entry point
//!
//! # Dependencies
//! `ascii-dag`, `ratatui`

use std::collections::HashMap;

use ascii_dag::LayoutConfig;
use ascii_dag::graph::{Graph, RenderMode};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use unicode_width::UnicodeWidthStr;

// ── Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Td,
    Lr,
    Bt,
    Rl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeShape {
    Rectangle,
    Rounded,
    Diamond,
    Circle,
    Asymmetric,
    Subroutine,
    Stadium,
    Hexagon,
}

#[derive(Debug, Clone)]
struct MermaidNode {
    label: String,
    shape: NodeShape,
}

#[derive(Debug, Clone)]
struct MermaidEdge {
    from: String,
    to: String,
    label: Option<String>,
}

#[derive(Debug)]
struct MermaidGraph {
    direction: Direction,
    nodes: HashMap<String, MermaidNode>,
    // Preserve insertion order for stable id assignment.
    nodeOrder: Vec<String>,
    edges: Vec<MermaidEdge>,
}

// ── Public entry point ─────────────────────────────────────────────────

/// Try to render mermaid source as a Unicode box-art diagram.
///
/// Returns `None` on parse failure, non-flowchart type, or empty graph,
/// signalling the caller to fall back to a plain code block.
///
/// Args:
///     code: Raw mermaid source (contents of a ` ```mermaid ` fence).
///
/// Returns:
///     Option<Vec<Vec<Span<'static>>>>: Styled lines for RenderedBlock::Code.
pub fn tryRenderMermaid(code: &str, availableWidth: usize) -> Option<Vec<Vec<Span<'static>>>> {
    let graph = parseMermaid(code)?;
    if graph.nodes.is_empty() {
        return None;
    }
    let lines = renderMermaid(&graph, availableWidth);
    if lines.is_empty() {
        return None;
    }
    Some(lines)
}

// ── Parser ─────────────────────────────────────────────────────────────

/// Parse mermaid flowchart source into an intermediate graph.
fn parseMermaid(code: &str) -> Option<MermaidGraph> {
    let mut lines = code.lines().peekable();

    // Skip blank lines and comments to find the header.
    let direction = loop {
        let line = lines.next()?.trim();
        if line.is_empty() || line.starts_with("%%") {
            continue;
        }
        break parseHeader(line)?;
    };

    let mut graph = MermaidGraph {
        direction,
        nodes: HashMap::new(),
        nodeOrder: Vec::new(),
        edges: Vec::new(),
    };

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("%%") {
            continue;
        }

        // Split on semicolons for multi-statement lines.
        for stmt in trimmed.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }
            parseStatement(&mut graph, stmt);
        }
    }

    Some(graph)
}

/// Parse the header line: `graph|flowchart [direction]`.
fn parseHeader(line: &str) -> Option<Direction> {
    let lower = line.to_lowercase();
    let rest = if let Some(r) = lower.strip_prefix("graph") {
        r
    } else if let Some(r) = lower.strip_prefix("flowchart") {
        r
    } else {
        return None;
    };

    let dirStr = rest.trim();
    let direction = match dirStr {
        "td" | "tb" | "" => Direction::Td,
        "lr" => Direction::Lr,
        "bt" => Direction::Bt,
        "rl" => Direction::Rl,
        _ => Direction::Td,
    };
    Some(direction)
}

/// Parse a single statement (node definition or edge chain).
fn parseStatement(graph: &mut MermaidGraph, stmt: &str) {
    // Try to parse as an edge chain first.
    // Edge chains: A --> B --> C or A -->|label| B
    if let Some(chain) = parseEdgeChain(stmt) {
        for segment in &chain {
            ensureNode(
                graph,
                &segment.fromId,
                &segment.fromLabel,
                segment.fromShape,
            );
            ensureNode(graph, &segment.toId, &segment.toLabel, segment.toShape);
            graph.edges.push(MermaidEdge {
                from: segment.fromId.clone(),
                to: segment.toId.clone(),
                label: segment.label.clone(),
            });
        }
        return;
    }

    // Fall back: standalone node definition.
    if let Some((id, label, shape)) = parseNodeSpec(stmt) {
        ensureNode(graph, &id, &Some(label), shape);
    }
}

/// Ensure a node exists in the graph, creating it if needed.
fn ensureNode(graph: &mut MermaidGraph, id: &str, label: &Option<String>, shape: NodeShape) {
    if graph.nodes.contains_key(id) {
        // Update label/shape if an explicit definition comes after implicit creation.
        if let Some(lbl) = label {
            let node = graph.nodes.get_mut(id).unwrap();
            node.label = lbl.clone();
            node.shape = shape;
        }
        return;
    }
    let label = label.clone().unwrap_or_else(|| id.to_string());
    graph.nodeOrder.push(id.to_string());
    graph
        .nodes
        .insert(id.to_string(), MermaidNode { label, shape });
}

// ── Edge chain parsing ─────────────────────────────────────────────────

struct EdgeSegment {
    fromId: String,
    fromLabel: Option<String>,
    fromShape: NodeShape,
    toId: String,
    toLabel: Option<String>,
    toShape: NodeShape,
    label: Option<String>,
}

/// Edge operators in order of longest match first.
const EDGE_OPS: &[&str] = &[
    "-.->", "==>", "-->", "---", "-..-", "-.-", "~~>", "~~~", "==",
];

/// Try to parse a statement as an edge chain (A --> B --> C).
fn parseEdgeChain(stmt: &str) -> Option<Vec<EdgeSegment>> {
    // Find the first edge operator to confirm this is an edge chain.
    let firstOp = EDGE_OPS.iter().find(|op| stmt.contains(**op))?;
    let _ = firstOp;

    // Tokenize: alternating node specs and edge operators with optional labels.
    let mut segments = Vec::new();
    let mut remaining = stmt.trim();

    // Parse first node.
    let (firstId, firstLabel, firstShape, rest) = parseNodeSpecFromStart(remaining)?;
    remaining = rest.trim();

    let mut prevId = firstId;
    let mut prevLabel = firstLabel;
    let mut prevShape = firstShape;

    while !remaining.is_empty() {
        // Find the edge operator.
        let (edgeLabel, afterOp) = parseEdgeOp(remaining)?;
        remaining = afterOp.trim();

        // Parse optional edge label: |text|
        let (label, afterLabel) = parseEdgeLabel(remaining, edgeLabel);
        remaining = afterLabel.trim();

        if remaining.is_empty() {
            break;
        }

        // Parse next node.
        let (nextId, nextLabel, nextShape, rest) = parseNodeSpecFromStart(remaining)?;
        remaining = rest.trim();

        segments.push(EdgeSegment {
            fromId: prevId.clone(),
            fromLabel: prevLabel.clone(),
            fromShape: prevShape,
            toId: nextId.clone(),
            toLabel: nextLabel.clone(),
            toShape: nextShape,
            label,
        });

        prevId = nextId;
        prevLabel = nextLabel;
        prevShape = nextShape;
    }

    if segments.is_empty() {
        None
    } else {
        Some(segments)
    }
}

/// Parse an edge operator from the start of the string.
/// Returns (label_from_op_text, remaining_string).
fn parseEdgeOp(s: &str) -> Option<(Option<String>, &str)> {
    for op in EDGE_OPS {
        if s.starts_with(op) {
            return Some((None, &s[op.len()..]));
        }
    }
    None
}

/// Parse optional `|label|` after an edge operator.
fn parseEdgeLabel<'a>(s: &'a str, opLabel: Option<String>) -> (Option<String>, &'a str) {
    if let Some(rest) = s.strip_prefix('|') {
        if let Some(endIdx) = rest.find('|') {
            let label = rest[..endIdx].trim().to_string();
            return (Some(label), &rest[endIdx + 1..]);
        }
    }
    (opLabel, s)
}

// ── Node spec parsing ──────────────────────────────────────────────────

/// Parse a node spec from the start of a string, returning (id, label, shape, remaining).
fn parseNodeSpecFromStart(s: &str) -> Option<(String, Option<String>, NodeShape, &str)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Parse the node id (alphanumeric + underscore + hyphen).
    let idEnd = s
        .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .unwrap_or(s.len());

    if idEnd == 0 {
        return None;
    }

    let id = s[..idEnd].to_string();
    let afterId = &s[idEnd..];

    // Try to parse shape delimiter.
    if let Some((label, shape, remaining)) = parseShapeDelimiter(afterId) {
        Some((id, Some(label), shape, remaining))
    } else {
        Some((id, None, NodeShape::Rectangle, afterId))
    }
}

/// Parse a standalone node spec: `id[label]` or bare `id`.
fn parseNodeSpec(s: &str) -> Option<(String, String, NodeShape)> {
    let (id, label, shape, _) = parseNodeSpecFromStart(s)?;
    let label = label.unwrap_or_else(|| id.clone());
    Some((id, label, shape))
}

/// Parse a shape delimiter from the start of a string.
/// Returns (label_text, shape, remaining_after_closer).
fn parseShapeDelimiter(s: &str) -> Option<(String, NodeShape, &str)> {
    // Order matters: check multi-char openers before single-char.
    let delimiters: &[(&str, &str, NodeShape)] = &[
        ("((", "))", NodeShape::Circle),
        ("[[", "]]", NodeShape::Subroutine),
        ("([", "])", NodeShape::Stadium),
        ("{{", "}}", NodeShape::Hexagon),
        ("[", "]", NodeShape::Rectangle),
        ("(", ")", NodeShape::Rounded),
        ("{", "}", NodeShape::Diamond),
        (">", "]", NodeShape::Asymmetric),
    ];

    for &(open, close, shape) in delimiters {
        if let Some(inner) = s.strip_prefix(open) {
            if let Some(endIdx) = inner.find(close) {
                let label = inner[..endIdx].trim();
                // Strip surrounding quotes if present.
                let label = label
                    .strip_prefix('"')
                    .and_then(|l| l.strip_suffix('"'))
                    .unwrap_or(label);
                let remaining = &inner[endIdx + close.len()..];
                return Some((label.to_string(), shape, remaining));
            }
        }
    }
    None
}

// ── Renderer ───────────────────────────────────────────────────────────

const DIAGRAM_STYLE: Style = Style::new().fg(Color::Cyan);

/// Render a parsed mermaid graph into styled spans via ascii-dag.
///
/// Centers the diagram horizontally if it fits within `availableWidth`.
fn renderMermaid(mermaid: &MermaidGraph, availableWidth: usize) -> Vec<Vec<Span<'static>>> {
    // Build id→index mapping for ascii-dag's usize node ids.
    let mut idMap: HashMap<&str, usize> = HashMap::new();
    let mut nodeEntries: Vec<(usize, String)> = Vec::new();
    for (idx, nodeId) in mermaid.nodeOrder.iter().enumerate() {
        idMap.insert(nodeId, idx);
        let node = &mermaid.nodes[nodeId];
        nodeEntries.push((idx, formatNodeLabel(node)));
    }

    // Build the ascii-dag graph.
    // NOTE: ascii-dag borrows &str from the node entries, so we build
    // the references from our owned strings.
    let nodeRefs: Vec<(usize, &str)> = nodeEntries
        .iter()
        .map(|(idx, label)| (*idx, label.as_str()))
        .collect();

    let mut dag = Graph::from_edges(&nodeRefs, &[]);

    // Collect edge labels so they live long enough for the borrow.
    let edgeLabels: Vec<Option<String>> = mermaid.edges.iter().map(|e| e.label.clone()).collect();

    // Add edges.
    for (i, edge) in mermaid.edges.iter().enumerate() {
        let fromIdx = match idMap.get(edge.from.as_str()) {
            Some(&idx) => idx,
            None => continue,
        };
        let toIdx = match idMap.get(edge.to.as_str()) {
            Some(&idx) => idx,
            None => continue,
        };
        // For BT/RL, reverse edge direction to simulate bottom-up / right-left.
        let (src, dst) = match mermaid.direction {
            Direction::Bt | Direction::Rl => (toIdx, fromIdx),
            _ => (fromIdx, toIdx),
        };
        dag.add_edge(src, dst, edgeLabels[i].as_deref());
    }

    let renderMode = match mermaid.direction {
        Direction::Td | Direction::Bt => RenderMode::Vertical,
        Direction::Lr | Direction::Rl => RenderMode::Horizontal,
    };
    dag.set_render_mode(renderMode);

    // Try render() first (cleaner output). Falls back to the layout
    // pipeline with CycleBreaking::DepthFirst for cyclic graphs.
    let output = {
        let attempt = dag.render();
        if attempt.contains("CYCLE DETECTED") {
            let mut config = LayoutConfig::fast();
            config.render_mode = renderMode;
            let ir = dag.compute_layout_with_config(&config);
            ir.render_scanline()
        } else {
            attempt
        }
    };

    // Strip trailing blank lines and common leading whitespace.
    let rawLines: Vec<&str> = output.lines().collect();

    // Trim trailing empty lines.
    let trimmedEnd = rawLines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .map_or(0, |i| i + 1);
    let lines = &rawLines[..trimmedEnd];

    // Trim leading empty lines.
    let trimmedStart = lines.iter().position(|l| !l.trim().is_empty()).unwrap_or(0);
    let lines = &lines[trimmedStart..];

    // Find minimum leading whitespace across non-empty lines and strip it.
    let minIndent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);

    // Strip leading indent and trailing whitespace from each line.
    let stripped: Vec<String> = lines
        .iter()
        .map(|line| {
            if line.len() > minIndent {
                line[minIndent..].trim_end().to_string()
            } else {
                line.trim().to_string()
            }
        })
        .collect();

    // Find the widest line in display columns (not bytes — Unicode box chars are multi-byte).
    let contentWidth = stripped
        .iter()
        .map(|l| UnicodeWidthStr::width(l.as_str()))
        .max()
        .unwrap_or(0);

    // Center if the diagram fits within the available width.
    let padding = if contentWidth < availableWidth {
        (availableWidth - contentWidth) / 2
    } else {
        0
    };
    let pad = " ".repeat(padding);

    stripped
        .iter()
        .map(|line| {
            if line.is_empty() {
                vec![Span::styled(String::new(), DIAGRAM_STYLE)]
            } else {
                vec![Span::styled(format!("{pad}{line}"), DIAGRAM_STYLE)]
            }
        })
        .collect()
}

/// Format a node label with shape indicators baked into the text.
fn formatNodeLabel(node: &MermaidNode) -> String {
    match node.shape {
        NodeShape::Rectangle => node.label.clone(),
        NodeShape::Rounded => format!("({})", node.label),
        NodeShape::Diamond => format!("<{}>", node.label),
        NodeShape::Circle => format!("(({}))", node.label),
        NodeShape::Asymmetric => format!(">{}", node.label),
        NodeShape::Subroutine => format!("[{}]", node.label),
        NodeShape::Stadium => format!("({})", node.label),
        NodeShape::Hexagon => format!("{{{}}}", node.label),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsesSimpleFlowchart() {
        let code = "graph TD\n    A[Start] --> B[End]";
        let graph = parseMermaid(code).unwrap();
        assert_eq!(graph.direction, Direction::Td);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.nodes["A"].label, "Start");
        assert_eq!(graph.nodes["B"].label, "End");
    }

    #[test]
    fn parsesEdgeLabels() {
        let code = "graph LR\n    A -->|yes| B";
        let graph = parseMermaid(code).unwrap();
        assert_eq!(graph.direction, Direction::Lr);
        assert_eq!(graph.edges[0].label.as_deref(), Some("yes"));
    }

    #[test]
    fn parsesChainedEdges() {
        let code = "graph TD\n    A --> B --> C";
        let graph = parseMermaid(code).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn parsesNodeShapes() {
        let code = "graph TD\n    A[rect]\n    B(rounded)\n    C{diamond}\n    D((circle))";
        let graph = parseMermaid(code).unwrap();
        assert_eq!(graph.nodes["A"].shape, NodeShape::Rectangle);
        assert_eq!(graph.nodes["B"].shape, NodeShape::Rounded);
        assert_eq!(graph.nodes["C"].shape, NodeShape::Diamond);
        assert_eq!(graph.nodes["D"].shape, NodeShape::Circle);
    }

    #[test]
    fn rejectsNonFlowchart() {
        assert!(parseMermaid("sequenceDiagram\n    A->>B: hello").is_none());
    }

    #[test]
    fn handlesImplicitNodes() {
        let code = "graph TD\n    A --> B";
        let graph = parseMermaid(code).unwrap();
        // Both A and B created implicitly with id as label.
        assert_eq!(graph.nodes["A"].label, "A");
        assert_eq!(graph.nodes["B"].label, "B");
    }

    #[test]
    fn skipsComments() {
        let code = "graph TD\n    %% this is a comment\n    A --> B";
        let graph = parseMermaid(code).unwrap();
        assert_eq!(graph.nodes.len(), 2);
    }

    #[test]
    fn handlesSemicolonSeparation() {
        let code = "graph TD\n    A --> B; B --> C";
        let graph = parseMermaid(code).unwrap();
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn rendersWithoutPanic() {
        let code =
            "graph TD\n    A[Start] --> B{Decision}\n    B -->|Yes| C[End]\n    B -->|No| D[Retry]";
        let result = tryRenderMermaid(code, 80);
        assert!(result.is_some());
        let lines = result.unwrap();
        assert!(!lines.is_empty());
    }

    #[test]
    fn rendersHorizontal() {
        let code = "graph LR\n    A --> B --> C";
        let result = tryRenderMermaid(code, 80);
        assert!(result.is_some());
    }

    #[test]
    fn emptyGraphReturnsNone() {
        let code = "graph TD";
        let result = tryRenderMermaid(code, 80);
        assert!(result.is_none());
    }

    #[test]
    fn rendersCyclicGraph() {
        let code = "graph TD\n    A --> B\n    B --> C\n    C --> A";
        let result = tryRenderMermaid(code, 80);
        assert!(
            result.is_some(),
            "cyclic graphs should render via back-edge reversal"
        );
        assert!(!result.unwrap().is_empty());
    }
}
