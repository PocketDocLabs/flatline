use crate::tool::EditOp;

use super::{expandFromAnchor, formatNumberedLines};

pub(in crate::tool) fn executeReadFile(
    path: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    anchor: Option<usize>,
) -> crate::message::Content {
    use base64::Engine;

    // File type detection via first 512 bytes.
    match std::fs::File::open(path) {
        Ok(mut file) => {
            use std::io::Read;
            let mut probe = [0u8; 512];
            let probeLen = match file.read(&mut probe) {
                Ok(n) => n,
                Err(e) => {
                    return crate::message::Content::text(format!("Failed to read file: {e}"));
                }
            };
            match classifyFile(&probe[..probeLen]) {
                FileKind::Image(fmt) => {
                    let fileSize = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    if fileSize > MAX_IMAGE_BYTES {
                        return crate::message::Content::text(format!(
                            "Image file ({fileSize} bytes). Too large to send inline \u{2014} maximum is 4 MB."
                        ));
                    }
                    match std::fs::read(path) {
                        Ok(bytes) => {
                            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                            let dataUri = format!("data:{};base64,{b64}", fmt.mimeType());
                            return crate::message::Content::withImages(
                                &format!("[{path}]"),
                                vec![dataUri],
                            );
                        }
                        Err(e) => {
                            return crate::message::Content::text(format!(
                                "Failed to read file: {e}"
                            ));
                        }
                    }
                }
                FileKind::Binary => {
                    return crate::message::Content::text(format!(
                        "Binary file ({} bytes). Use shell tools to inspect.",
                        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
                    ));
                }
                FileKind::Text => {}
            }
        }
        Err(e) => return crate::message::Content::text(format!("Failed to read file: {e}")),
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return crate::message::Content::text(format!("Failed to read file: {e}")),
    };

    // Anchor mode: expand from a line based on indentation.
    if let Some(anchorLine) = anchor {
        return crate::message::Content::text(expandFromAnchor(&content, anchorLine));
    }

    crate::message::Content::text(formatNumberedLines(&content, offset, limit))
}

pub(super) fn executeWriteFile(path: &str, content: &str) -> String {
    // Create parent directories if needed.
    if let Some(parent) = std::path::Path::new(path).parent()
        && !parent.exists()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("Failed to create directories: {e}");
    }
    match std::fs::write(path, content) {
        Ok(()) => {
            tracing::debug!(%path, bytes = content.len(), "wrote file");
            format!("Wrote {} bytes to {path}", content.len())
        }
        Err(e) => {
            tracing::warn!(%path, error = %e, "writeFile failed");
            format!("Failed to write file: {e}")
        }
    }
}

pub(super) fn executeEditFile(
    path: &str,
    oldString: &str,
    newString: &str,
    replaceAll: bool,
) -> String {
    if oldString == newString {
        return "old_string and new_string are identical.".into();
    }
    if oldString.is_empty() {
        return "old_string cannot be empty.".into();
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read file: {e}"),
    };

    let matchCount = content.matches(oldString).count();

    if matchCount == 0 {
        return "No match found for old_string.".into();
    }

    if !replaceAll && matchCount > 1 {
        return format!(
            "Found {matchCount} matches for old_string. \
             Provide more context to make a unique match, or set replace_all to true."
        );
    }

    let newContent = if replaceAll {
        content.replace(oldString, newString)
    } else {
        content.replacen(oldString, newString, 1)
    };

    match std::fs::write(path, &newContent) {
        Ok(()) => {
            if replaceAll && matchCount > 1 {
                tracing::debug!(%path, count = matchCount, "editFile replaced all occurrences");
                format!("Replaced {matchCount} occurrences in {path}.")
            } else {
                tracing::debug!(%path, "editFile applied");
                format!("Applied edit to {path}.")
            }
        }
        Err(e) => {
            tracing::warn!(%path, error = %e, "editFile write failed");
            format!("Failed to write file: {e}")
        }
    }
}

pub(super) fn executeMultiEdit(path: &str, edits: &[EditOp]) -> String {
    if edits.is_empty() {
        return "No edits provided.".into();
    }

    let original = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("Failed to read file: {e}"),
    };

    let mut content = original;

    // Validate and apply each edit sequentially against the in-memory copy.
    for (i, edit) in edits.iter().enumerate() {
        if edit.oldString.is_empty() {
            return format!(
                "Edit {}: old_string cannot be empty. No edits were applied.",
                i + 1
            );
        }
        if edit.oldString == edit.newString {
            return format!(
                "Edit {}: old_string and new_string are identical. No edits were applied.",
                i + 1
            );
        }

        let matchCount = content.matches(&edit.oldString).count();

        if matchCount == 0 {
            return format!(
                "Edit {}: no match found for old_string. No edits were applied.",
                i + 1
            );
        }

        if !edit.replaceAll && matchCount > 1 {
            return format!(
                "Edit {}: found {matchCount} matches for old_string. \
                 Provide more context or set replace_all. No edits were applied.",
                i + 1
            );
        }

        content = if edit.replaceAll {
            content.replace(&edit.oldString, &edit.newString)
        } else {
            content.replacen(&edit.oldString, &edit.newString, 1)
        };
    }

    // All edits validated — write once.
    match std::fs::write(path, &content) {
        Ok(()) => {
            tracing::debug!(%path, edits = edits.len(), "multiEdit applied");
            format!("Applied {} edits to {path}.", edits.len())
        }
        Err(e) => {
            tracing::warn!(%path, error = %e, "multiEdit write failed");
            format!("Failed to write file: {e}")
        }
    }
}

pub(super) fn executeCopyFile(src: &str, dest: &str, overwrite: bool) -> String {
    let srcPath = std::path::Path::new(src);
    let destPath = std::path::Path::new(dest);
    if !srcPath.exists() {
        return format!("Source does not exist: {src}");
    }
    if destPath.exists() && !overwrite {
        return format!("Destination already exists: {dest}. Set overwrite=true to replace.");
    }
    if let Some(parent) = destPath.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("Failed to create parent directories of {dest}: {e}");
    }
    if srcPath.is_dir() {
        match copyDirRecursive(srcPath, destPath) {
            Ok(()) => {
                tracing::debug!(%src, %dest, "copied directory");
                format!("Copied directory {src} \u{2192} {dest}.")
            }
            Err(e) => {
                tracing::warn!(%src, %dest, error = %e, "copyFile dir failed");
                format!("Failed to copy directory: {e}")
            }
        }
    } else {
        match std::fs::copy(srcPath, destPath) {
            Ok(bytes) => {
                tracing::debug!(%src, %dest, bytes, "copied file");
                format!("Copied {src} \u{2192} {dest} ({bytes} bytes).")
            }
            Err(e) => {
                tracing::warn!(%src, %dest, error = %e, "copyFile failed");
                format!("Failed to copy file: {e}")
            }
        }
    }
}

fn copyDirRecursive(src: &std::path::Path, dest: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if kind.is_dir() {
            copyDirRecursive(&from, &to)?;
        } else if kind.is_symlink() {
            // Reproduce symlinks by reading the target.
            let target = std::fs::read_link(&from)?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &to)?;
            #[cfg(windows)]
            {
                if target.is_dir() {
                    std::os::windows::fs::symlink_dir(&target, &to)?;
                } else {
                    std::os::windows::fs::symlink_file(&target, &to)?;
                }
            }
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

pub(super) fn executeMoveFile(src: &str, dest: &str, overwrite: bool) -> String {
    let srcPath = std::path::Path::new(src);
    let destPath = std::path::Path::new(dest);
    if !srcPath.exists() {
        return format!("Source does not exist: {src}");
    }
    if destPath.exists() && !overwrite {
        return format!("Destination already exists: {dest}. Set overwrite=true to replace.");
    }
    if let Some(parent) = destPath.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("Failed to create parent directories of {dest}: {e}");
    }
    // Try a rename first (cheap, atomic when on the same filesystem). Fall back
    // to copy+delete on EXDEV (cross-device link) or other rename failures.
    match std::fs::rename(srcPath, destPath) {
        Ok(()) => {
            tracing::debug!(%src, %dest, "moved file via rename");
            format!("Moved {src} \u{2192} {dest}.")
        }
        Err(e) => {
            tracing::debug!(%src, %dest, error = %e, "rename failed, falling back to copy+delete");
            let copyResult = if srcPath.is_dir() {
                copyDirRecursive(srcPath, destPath)
            } else {
                std::fs::copy(srcPath, destPath).map(|_| ())
            };
            if let Err(e) = copyResult {
                tracing::warn!(%src, %dest, error = %e, "moveFile copy fallback failed");
                return format!("Failed to move (copy phase): {e}");
            }
            let removeResult = if srcPath.is_dir() {
                std::fs::remove_dir_all(srcPath)
            } else {
                std::fs::remove_file(srcPath)
            };
            match removeResult {
                Ok(()) => {
                    tracing::debug!(%src, %dest, "moved file via copy+delete fallback");
                    format!("Moved {src} \u{2192} {dest} (cross-device, copy+delete).")
                }
                Err(e) => {
                    tracing::warn!(%src, %dest, error = %e, "moveFile remove-source fallback failed");
                    format!("Copied {src} \u{2192} {dest} but failed to remove source: {e}")
                }
            }
        }
    }
}

pub(super) fn executeDeleteFile(path: &str, recursive: bool) -> String {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return format!("Path does not exist: {path}");
    }
    if p.is_dir() {
        if recursive {
            match std::fs::remove_dir_all(p) {
                Ok(()) => {
                    tracing::info!(%path, "deleted directory tree");
                    format!("Deleted directory tree {path}.")
                }
                Err(e) => {
                    tracing::warn!(%path, error = %e, "deleteFile dir failed");
                    format!("Failed to delete directory: {e}")
                }
            }
        } else {
            match std::fs::remove_dir(p) {
                Ok(()) => {
                    tracing::info!(%path, "deleted empty directory");
                    format!("Deleted empty directory {path}.")
                }
                Err(e) => {
                    tracing::warn!(%path, error = %e, "deleteFile dir failed (not empty)");
                    format!(
                        "Failed to delete directory: {e}. Set recursive=true to delete contents."
                    )
                }
            }
        }
    } else {
        match std::fs::remove_file(p) {
            Ok(()) => {
                tracing::info!(%path, "deleted file");
                format!("Deleted {path}.")
            }
            Err(e) => {
                tracing::warn!(%path, error = %e, "deleteFile failed");
                format!("Failed to delete file: {e}")
            }
        }
    }
}

pub(super) fn executeMakeDirs(path: &str) -> String {
    match std::fs::create_dir_all(path) {
        Ok(()) => {
            tracing::debug!(%path, "created directory");
            format!("Created directory {path}.")
        }
        Err(e) => {
            tracing::warn!(%path, error = %e, "makeDirs failed");
            format!("Failed to create directory: {e}")
        }
    }
}

/// Classification of a file's content type based on magic bytes.
pub(in crate::tool) enum FileKind {
    Text,
    Image(ImageFormat),
    Binary,
}

/// Recognized image formats (by magic bytes).
pub(in crate::tool) enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Bmp,
    Webp,
}

impl ImageFormat {
    fn mimeType(&self) -> &'static str {
        match self {
            ImageFormat::Png => "image/png",
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Gif => "image/gif",
            ImageFormat::Bmp => "image/bmp",
            ImageFormat::Webp => "image/webp",
        }
    }
}

/// Maximum image file size for inline base64 encoding (4 MB).
const MAX_IMAGE_BYTES: u64 = 4 * 1024 * 1024;

/// Classify file content by probing magic bytes.
pub(in crate::tool) fn classifyFile(bytes: &[u8]) -> FileKind {
    if bytes.is_empty() {
        return FileKind::Text;
    }

    // Image signatures — check before generic binary.
    if bytes.starts_with(b"\x89PNG") {
        return FileKind::Image(ImageFormat::Png);
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return FileKind::Image(ImageFormat::Jpeg);
    }
    if bytes.starts_with(b"GIF8") {
        return FileKind::Image(ImageFormat::Gif);
    }
    if bytes.starts_with(b"BM") {
        return FileKind::Image(ImageFormat::Bmp);
    }
    // WebP: starts with RIFF....WEBP.
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return FileKind::Image(ImageFormat::Webp);
    }

    // Non-image binary signatures.
    const BINARY_MAGIC: &[&[u8]] = &[
        b"PK\x03\x04",       // ZIP/DOCX/JAR
        b"\x7fELF",          // ELF
        b"\xfe\xed\xfa",     // Mach-O
        b"\xcf\xfa\xed\xfe", // Mach-O (reversed)
        b"%PDF",             // PDF
        b"\x1f\x8b",         // gzip
    ];
    for sig in BINARY_MAGIC {
        if bytes.starts_with(sig) {
            return FileKind::Binary;
        }
    }

    // NUL byte check (strong binary indicator in first 512 bytes).
    if bytes.contains(&0x00) {
        return FileKind::Binary;
    }

    FileKind::Text
}
