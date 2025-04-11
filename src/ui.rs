use eframe::egui;

pub fn create_image_from_rgba(rgba: &[u8], width: usize, height: usize) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied([width, height], rgba)
}
