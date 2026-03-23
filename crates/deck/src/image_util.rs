//! Image encoding utilities for clipboard paste.
//!
//! Converts raw RGBA pixel data (from arboard clipboard) to PNG bytes.
//! Resizes oversized images to fit within a 2048x2048 bounding box.
//!
//! # Public API
//! - [`encodeRgbaToPng`] — RGBA pixels to PNG bytes
//!
//! # Dependencies
//! `image`

/// Maximum dimension (width or height) before resizing.
const MAX_DIMENSION: u32 = 2048;

/// Encode raw RGBA pixel data to PNG bytes.
///
/// Args:
///     rgba: Raw RGBA pixel data (4 bytes per pixel).
///     width: Image width in pixels.
///     height: Image height in pixels.
///
/// Returns:
///     Vec<u8>: PNG-encoded bytes.
pub fn encodeRgbaToPng(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_raw(width, height, rgba.to_vec())
        .expect("RGBA buffer size mismatch");

    let dynamic = image::DynamicImage::ImageRgba8(img);

    // Resize if either dimension exceeds the limit.
    let resized = if width > MAX_DIMENSION || height > MAX_DIMENSION {
        dynamic.resize(MAX_DIMENSION, MAX_DIMENSION, image::imageops::FilterType::Lanczos3)
    } else {
        dynamic
    };

    let mut buf = Vec::new();
    resized
        .write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Png,
        )
        .expect("PNG encoding failed");
    buf
}
