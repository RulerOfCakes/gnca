use std::collections::HashMap;

use eframe::egui::{self};
use eframe::{NativeOptions, egui::ViewportBuilder, run_native};
use gnca::imageutils::load_emoji;
use gnca::ui::create_image_from_rgba;
fn main() -> eframe::Result {
    unsafe {
        std::env::set_var("WAYLAND_DISPLAY", ""); // Force X11 on Linux
    }
    tracing_subscriber::fmt::init();
    let options = NativeOptions {
        viewport: ViewportBuilder::default().with_inner_size([1200., 800.]),
        ..Default::default()
    };
    run_native(
        "Growing Neural Cellular Automata",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            let mut app = MyApp::default();
            let smiling_emoji_bytes = load_emoji("😊", 40, 40).unwrap();
            let uri = "bytes://smiling_emoji.png";
            app.insert_or_replace_image(
                &cc.egui_ctx,
                uri,
                create_image_from_rgba(&smiling_emoji_bytes, 40, 40),
            );
            Ok(Box::new(app))
        }),
    )
}

#[derive(Default, Debug)]
enum AppState {
    #[default]
    Idle,
    Training,
    Inference,
}

#[derive(Default)]
struct MyApp {
    image_handles: HashMap<String, eframe::egui::TextureHandle>,
    state: AppState,
}

impl MyApp {
    fn insert_or_replace_image(
        &mut self,
        ctx: &egui::Context,
        name: &str,
        image: egui::ColorImage,
    ) {
        let handle = ctx.load_texture(name, image, eframe::egui::TextureOptions::default());
        self.image_handles.insert(name.to_string(), handle);
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::panel::SidePanel::left("configuration_panel")
                .default_width(300.0)
                .show_inside(ui, |ui| {
                    ui.label("Model Parameters");
                    ui.horizontal(|ui| {
                        ui.label("Grid Size:");
                        ui.add(egui::DragValue::new(&mut 40));
                    });
                    ui.horizontal(|ui| {
                        ui.label("State Channels:");
                        ui.add(egui::DragValue::new(&mut 16));
                    });
                    ui.separator();
                    ui.label("Training Options");
                    match self.state {
                        AppState::Idle => {
                            ui.horizontal(|ui| {
                                if ui.button("Start Training").clicked() {
                                    self.state = AppState::Training;
                                }
                                if ui.button("Start Inference").clicked() {
                                    self.state = AppState::Inference;
                                }
                            });
                        }
                        AppState::Training => {
                            ui.label("Training in progress...");
                            if ui.button("Stop Training").clicked() {
                                self.state = AppState::Idle;
                            }
                        }
                        AppState::Inference => {
                            ui.label("Inference in progress...");
                            if ui.button("Stop Inference").clicked() {
                                self.state = AppState::Idle;
                            }
                        }
                    }
                    ui.separator();
                    ui.label("Visualization Options");
                });
            egui::ScrollArea::both().show(ui, |ui| {
                for (name, handle) in &self.image_handles {
                    ui.label(name);
                    ui.image(handle);
                }
            });
        });
    }
}
