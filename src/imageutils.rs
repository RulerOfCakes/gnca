use std::io::Cursor;

use image::ImageReader;
use tracing::info;

const IMAGE_SIZE_LIMIT: u64 = 1024 * 1024 * 10; // 10 MB

pub fn load_image_from_url(
    image_url: &str,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, anyhow::Error> {
    info!("Loading image from URL: {}", image_url);
    let mut response = ureq::get(image_url)
        .call()
        .expect("Failed to load image from URL");
    let image_bytes = response
        .body_mut()
        .with_config()
        .limit(IMAGE_SIZE_LIMIT)
        .read_to_vec()
        .expect("Failed to read image bytes");

    let reader = ImageReader::new(Cursor::new(image_bytes)).with_guessed_format()?;
    let image = reader.decode()?;

    let image = image.thumbnail(width, height);
    let rgb_image = image.to_rgba8();

    Ok(rgb_image.to_vec())
}

fn expand_emoji(emoji: &str) -> String {
    let emoji_unicode: Vec<u32> = emoji.chars().map(|c| c as u32).collect();
    let emoji_code: String = emoji_unicode
        .iter()
        .map(|c| format!("{:x}", c))
        .collect::<Vec<String>>()
        .join("_");
    emoji_code
}

pub fn load_emoji(emoji: &str, width: u32, height: u32) -> Result<Vec<u8>, anyhow::Error> {
    let emoji_code = expand_emoji(emoji);
    let url = format!(
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{}.png?raw=true",
        emoji_code
    );
    load_image_from_url(&url, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_emoji() {
        let emoji = "😀";
        let expanded = expand_emoji(emoji);
        assert_eq!(expanded, "1f600");

        let emoji = "🏌🏿";
        let expanded = expand_emoji(emoji);
        assert_eq!(expanded, "1f3cc_1f3ff");
    }

    #[test]
    fn test_load_emoji() {
        let emoji = "😀";
        let width = 128;
        let height = 128;

        let result = load_emoji(emoji, width, height);
        assert!(result.is_ok());
        let image_data = result.unwrap();
        assert!(!image_data.is_empty());
        assert!(image_data.iter().any(|&x| x != 0));
    }
}
