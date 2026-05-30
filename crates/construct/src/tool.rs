//! Tool definitions and execution.
//!
//! Defines the tools available to the agent and handles execution.
//! Shell commands go through the construct-owned Shell session.
//! File operations run in-process. Search/diff tools use subprocesses
//! (rg, sg, git) to keep work off the shared terminal.
//!
//! # Public API
//! - [`parse`] - parse a tool call into a typed action
//! - [`summarize`] - format a short tool action summary
//!
//! # Dependencies
//! `serde_json`, `regex`, `similar`, `tokio::process`

mod action;
mod defs;
mod execute;
mod parse;
mod presentation;

pub use action::{EditOp, ShellImpact, ToolAction, ToolParseError, ToolSet};
pub(crate) use action::{
    filterDefs, needsJobPlane, needsLsp, needsMcp, needsMonitor, needsRegistry, needsTask,
    needsTranscript, needsWakes, needsWeb,
};
pub(crate) use defs::{
    addPermissionEscalationFieldsToDefs, builtinDefs, builtinDefsWithPermissionEscalation,
    stripPermissionEscalationArgs, stripPermissionEscalationObject,
};
pub(crate) use execute::execute;
pub(crate) use execute::truncateOutput;
pub use parse::parse;
pub use presentation::summarize;
pub(crate) use presentation::{diffPreview, proposedContent};

#[cfg(test)]
use self::execute::{FileKind, ImageFormat, SubprocessError, classifyFile, executeReadFile};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parseReportsMissingRequiredFieldStructurally() {
        let err = parse("readFile", "{}").unwrap_err();
        assert_eq!(err, ToolParseError::MissingField { field: "path" });
        assert_eq!(err.to_string(), "Missing required field 'path'.");
    }

    #[test]
    fn parseReportsWrongFieldTypeStructurally() {
        let err = parse("readFile", r#"{"path":"src/main.rs","offset":"1"}"#).unwrap_err();
        assert_eq!(
            err,
            ToolParseError::WrongType {
                field: "offset",
                expected: "integer",
            }
        );
        assert_eq!(err.to_string(), "Field 'offset': expected integer.");
    }

    #[test]
    fn parseReportsMalformedJsonStructurally() {
        let err = parse("readFile", r#"{"path":"src/main.rs""#).unwrap_err();
        assert!(matches!(err, ToolParseError::MalformedJson(_)));
        assert!(err.to_string().starts_with("Malformed JSON arguments:"));
    }

    #[test]
    fn parseReportsInvalidImpactStructurally() {
        let err = parse(
            "shell",
            r#"{"command":"pwd","explanation":"inspect cwd","impact":"tiny"}"#,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ToolParseError::InvalidField {
                field: "impact",
                expected: "one of: read, minorMod, majorMod, delete",
            }
        );
        assert_eq!(
            err.to_string(),
            "Field 'impact': expected one of: read, minorMod, majorMod, delete.",
        );
    }

    #[test]
    fn parseReportsNestedEditFieldStructurally() {
        let err = parse("multiEdit", r#"{"path":"src/main.rs","edits":[{}]}"#).unwrap_err();
        assert_eq!(
            err,
            ToolParseError::MissingNestedField {
                context: "Edit",
                field: "old_string",
            }
        );
        assert_eq!(err.to_string(), "Edit missing 'old_string'.");
    }

    #[test]
    fn parseAllowsEmptyReplacementString() {
        let action = parse(
            "editFile",
            r#"{"path":"src/main.rs","old_string":"delete me","new_string":""}"#,
        )
        .unwrap();
        assert!(matches!(
            action,
            ToolAction::EditFile { newString, .. } if newString.is_empty()
        ));
    }

    #[test]
    fn parseReportsNestedEditWrongTypeStructurally() {
        let err = parse(
            "multiEdit",
            r#"{"path":"src/main.rs","edits":[{"old_string":"a","new_string":"b","replace_all":"yes"}]}"#,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ToolParseError::WrongNestedType {
                context: "Edit",
                field: "replace_all",
                expected: "boolean",
            }
        );
        assert_eq!(
            err.to_string(),
            "Edit field 'replace_all': expected boolean.",
        );
    }

    #[test]
    fn parseReportsInvalidEnumLikeFieldsStructurally() {
        let err = parse("grep", r#"{"pattern":"TODO","output_mode":"everything"}"#).unwrap_err();
        assert_eq!(
            err,
            ToolParseError::InvalidField {
                field: "output_mode",
                expected: "one of: files, content, count",
            }
        );

        let err = parse("diagnostics", r#"{"path":"src/main.rs","severity":"info"}"#).unwrap_err();
        assert_eq!(
            err,
            ToolParseError::InvalidField {
                field: "severity",
                expected: "one of: error, warning",
            }
        );
    }

    #[test]
    fn parseReportsStringArrayTypeStructurally() {
        let err = parse(
            "webSearch",
            r#"{"query":"rust","allowed_domains":["docs.rs", 42]}"#,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ToolParseError::WrongType {
                field: "allowed_domains",
                expected: "array of strings",
            }
        );
        assert_eq!(
            err.to_string(),
            "Field 'allowed_domains': expected array of strings.",
        );
    }

    #[test]
    fn subprocessErrorsFormatAtToolBoundary() {
        let err = SubprocessError::Timeout {
            program: "rg".into(),
            seconds: 30,
        };
        assert_eq!(err.to_string(), "rg timed out after 30s.");

        let err = SubprocessError::Failed {
            program: "git".into(),
            status: "exit status: 2".into(),
            output: "fatal: bad revision\n".into(),
        };
        assert_eq!(
            err.to_string(),
            "git failed (exit exit status: 2): fatal: bad revision",
        );
    }

    #[test]
    fn classifyPng() {
        let header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Png) => {}
            other => panic!("expected PNG, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyJpeg() {
        let header = b"\xff\xd8\xff\xe0\x00\x10JFIF";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Jpeg) => {}
            other => panic!("expected JPEG, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyGif() {
        let header = b"GIF89a\x01\x00\x01\x00";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Gif) => {}
            other => panic!("expected GIF, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyWebp() {
        let header = b"RIFF\x00\x00\x00\x00WEBPVP8 ";
        match classifyFile(header) {
            FileKind::Image(ImageFormat::Webp) => {}
            other => panic!("expected WebP, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyElf() {
        let header = b"\x7fELF\x02\x01\x01\x00";
        match classifyFile(header) {
            FileKind::Binary => {}
            other => panic!("expected Binary, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyPlainText() {
        let header = b"fn main() {\n    println!(\"hello\");\n}";
        match classifyFile(header) {
            FileKind::Text => {}
            other => panic!("expected Text, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyNulAsBinary() {
        let header = b"some text\x00more text";
        match classifyFile(header) {
            FileKind::Binary => {}
            other => panic!("expected Binary, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn classifyEmptyAsText() {
        match classifyFile(b"") {
            FileKind::Text => {}
            other => panic!("expected Text, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn readFileTextReturnsContent() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "line 1\nline 2\n").unwrap();
        let result = executeReadFile(tmp.path().to_str().unwrap(), None, None, None);
        let text = result.textContent();
        assert!(text.contains("line 1"));
        assert!(text.contains("line 2"));
        assert!(!result.hasImages());
    }

    #[test]
    fn readFilePngReturnsImageContent() {
        // Write a minimal 1x1 PNG.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let png: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
            0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
            0xf8, 0xcf, 0xc0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc, 0x33, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        std::fs::write(tmp.path(), png).unwrap();
        let result = executeReadFile(tmp.path().to_str().unwrap(), None, None, None);
        assert!(result.hasImages());
        let uris = result.imageUris();
        assert_eq!(uris.len(), 1);
        assert!(uris[0].starts_with("data:image/png;base64,"));
    }

    #[test]
    fn readFileBinaryReturnsError() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let elf = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        std::fs::write(tmp.path(), elf).unwrap();
        let result = executeReadFile(tmp.path().to_str().unwrap(), None, None, None);
        assert!(!result.hasImages());
        assert!(result.textContent().contains("Binary file"));
    }
}
