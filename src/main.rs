use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::config::Config;
use burn::optim::AdamConfig;
use eframe::egui::{self};
use eframe::{NativeOptions, egui::ViewportBuilder, run_native};
use gnca::imageutils::{RgbaImageBuffer, load_emoji};
use gnca::model::original_model::{ModelType, OriginalModel, OriginalModelConfig};
use gnca::model::train::{TrainingConfig, TrainingFeedback};
use gnca::ui::{add_draggable_option, create_image_from_rgba};

const ARTIFACT_PATH: &str = "artifacts";
const TRAINING_CFG_PATH: &str = "training_cfg.json";

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
            let mut app = MyApp::new();
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
    Training {
        image_recv: Receiver<TrainingFeedback<WgpuAutodiff>>,
        interrupt_send: Sender<()>,
        handle: Option<JoinHandle<OriginalModel<WgpuAutodiff>>>,
    },
    Inference {
        image_recv: Receiver<TrainingFeedback<WgpuAutodiff>>, // TODO: fix
        interrupt_send: Sender<()>,
    },
}

type WgpuBackend = Wgpu<f32, i32>;
type WgpuAutodiff = Autodiff<WgpuBackend>;

struct MyApp {
    image_handles: HashMap<String, eframe::egui::TextureHandle>,
    state: AppState,
    training_cfg: TrainingConfig,
    target_image: Arc<RgbaImageBuffer>, // TODO: let this be modifiable from ui
    device: WgpuDevice,
    model: Option<OriginalModel<WgpuAutodiff>>, // burn models are Send but not Sync, therefore needs to be moved later on
}

impl MyApp {
    fn new() -> Self {
        let training_cfg = TrainingConfig::load(ARTIFACT_PATH.to_owned() + "/" + TRAINING_CFG_PATH)
            .unwrap_or(TrainingConfig::new(
                OriginalModelConfig::new(ModelType::Growing),
                AdamConfig::new(),
            ));

        Self {
            image_handles: HashMap::new(),
            state: AppState::Idle,
            training_cfg,
            target_image: Arc::new(load_emoji("😊", 40, 40).unwrap()),
            device: WgpuDevice::default(),
            model: None,
        }
    }
    fn insert_or_replace_image(
        &mut self,
        ctx: &egui::Context,
        name: &str,
        image: egui::ColorImage,
    ) {
        let handle = ctx.load_texture(name, image, eframe::egui::TextureOptions::default());
        self.image_handles.insert(name.to_string(), handle);
    }

    fn train_model(&mut self) {
        if !matches!(self.state, AppState::Idle) {
            return;
        }
        let image = self.target_image.clone();
        let device = self.device.clone();
        let training_cfg = self.training_cfg.clone();

        // Communication channels for feedback and interrupts
        let (feedback_send, feedback_recv) = channel();
        let (int_send, int_recv) = channel();

        let handle = thread::spawn(move || {
            training_cfg.train::<WgpuAutodiff>(
                ARTIFACT_PATH,
                &image,
                device,
                Some(feedback_send),
                Some(int_recv),
            )
        });

        self.state = AppState::Training {
            image_recv: feedback_recv,
            interrupt_send: int_send,
            handle: Some(handle),
        };
    }

    fn update_state(&mut self) {
        match &mut self.state {
            AppState::Idle => return,
            AppState::Training {
                image_recv,
                handle: handle_opt,
                ..
            } => {
                match image_recv.try_recv() {
                    Ok(feedback) => {
                        let (regression, target) = feedback;
                        todo!();
                    }
                    Err(_e) => {}
                }
                if let Some(handle) = handle_opt.take() {
                    if handle.is_finished() {
                        let model = handle.join().unwrap();
                        self.model = Some(model);
                        self.state = AppState::Idle;
                    }
                }
            }
            AppState::Inference { image_recv, .. } => {
                todo!();
                match image_recv.try_recv() {
                    Ok(feedback) => {
                        let (regression, target) = feedback;
                    }
                    Err(e) => {
                        if e == std::sync::mpsc::TryRecvError::Disconnected {
                            // detach the receiver
                            self.state = AppState::Idle;
                        }
                    }
                }
            }
        }
        todo!("Upate images");
    }

    fn infer_model(&mut self) {
        todo!()
    }

    fn stop_model(&mut self) {
        match &mut self.state {
            AppState::Training { interrupt_send, .. } => {
                let _ = interrupt_send.send(());
            }
            AppState::Inference { interrupt_send, .. } => {
                let _ = interrupt_send.send(());
            }
            _ => {}
        };
        // Simply dropping the previous state is enough
        self.state = AppState::Idle;
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        self.update_state();
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::panel::SidePanel::left("configuration_panel")
                .default_width(300.0)
                .show_inside(ui, |ui| {
                    ui.heading("Model Parameters");
                    add_draggable_option(
                        ui,
                        "Grid Size",
                        Some("The size of the cellular automata grid."),
                        &mut self.training_cfg.model.target_size,
                        None,
                    );
                    add_draggable_option(
                        ui,
                        "State Channels",
                        Some("The number of channels to represent each cell's state."),
                        &mut self.training_cfg.model.state_channels,
                        None,
                    );
                    add_draggable_option(ui,
                    "Target Padding",
                    Some("Padding for the target image."),
                    &mut self.training_cfg.model.target_padding,
                    None);
                    add_draggable_option(
                        ui,
                        "Pool Size",
                        Some("Size of the pool for the cellular automata."),
                        &mut self.training_cfg.pool_size,
                        None,
                    );
                    add_draggable_option(
                        ui,
                        "Cell Fire Rate",
                        Some("Rate of cells chosen for stochastic update."),
                        &mut self.training_cfg.model.cell_fire_rate,
                        Some(0.01),
                    );
                    add_draggable_option(
                        ui,
                        "Alive Threshold",
                        Some("Threshold for a cell to be considered alive."),
                        &mut self.training_cfg.model.alive_threshold,
                        Some(0.01),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Model Type");
                        ui.selectable_value(
                            &mut self.training_cfg.model.model_type,
                            ModelType::Growing,
                            "Growing",
                        );
                        ui.selectable_value(
                            &mut self.training_cfg.model.model_type,
                            ModelType::Persistent,
                            "Persistent",
                        );
                        ui.selectable_value(
                            &mut self.training_cfg.model.model_type,
                            ModelType::Regenerating,
                            "Regenerating",
                        );
                    });
                    ui.separator();
                    ui.heading("Training Parameters");
                    add_draggable_option(
                        ui,
                        "Num Workers",
                        Some(
                            "Number of worker threads for data loading.
                        TODO: this will also be applied to the training procedure itself.",
                        ),
                        &mut self.training_cfg.num_workers,
                        None,
                    );
                    add_draggable_option(
                        ui,
                        "Batch Size",
                        Some("Batch size for each training iteration."),
                        &mut self.training_cfg.batch_size,
                        None,
                    );
                    add_draggable_option(
                        ui,
                        "Num Epochs",
                        Some("Number of epochs to train the model."),
                        &mut self.training_cfg.num_epochs,
                        None,
                    );
                    add_draggable_option(
                        ui,
                        "Learning Rate",
                        Some("Learning rate for the optimizer."),
                        &mut self.training_cfg.lr,
                        Some(0.001),
                    );
                    add_draggable_option(
                        ui,
                        "Seed",
                        Some("Set seed for deterministic behavior. Note that this is also dependent on the platform backend."),
                        &mut self.training_cfg.seed,
                        None,
                    );
                    // TODO: add Adam config(they are private)
                    match &self.state {
                        AppState::Idle => {
                            ui.horizontal(|ui| {
                                if ui.button("Start Training").clicked() {
                                    self.train_model();
                                }
                                if ui.button("Start Inference").clicked() {
                                    self.infer_model();
                                }
                            });
                        }
                        AppState::Training {..} => {
                            ui.label("Training in progress...");
                            if ui.button("Stop Training").clicked() {
                                self.stop_model();
                            }
                        }
                        AppState::Inference {..} => {
                            ui.label("Inference in progress...");
                            if ui.button("Stop Inference").clicked() {
                                self.stop_model();
                            }
                        }
                    }
                    ui.separator();
                    ui.heading("Visualization Options");
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
