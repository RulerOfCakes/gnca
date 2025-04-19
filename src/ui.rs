use eframe::egui;

pub fn create_image_from_rgba(rgba: &[u8], width: usize, height: usize) -> egui::ColorImage {
    egui::ColorImage::from_rgba_unmultiplied([width, height], rgba)
}

pub fn add_draggable_option<T: eframe::emath::Numeric>(
    ui: &mut egui::Ui,
    label: &str,
    description: Option<&str>,
    reference: &mut T,
    speed: Option<f64>,
) -> egui::InnerResponse<egui::Response> {
    ui.horizontal(|ui| {
        let label = ui.label(label);
        if let Some(description) = description {
            label.on_hover_text(description);
        }
        ui.add(egui::DragValue::new(reference).speed(speed.unwrap_or(0.25)))
    })
}
