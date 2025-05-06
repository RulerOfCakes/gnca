use std::sync::mpsc::Sender;

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{ElementConversion, Tensor, backend::AutodiffBackend},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing::{Level, info, span, warn};

use crate::{imageutils::RgbaImageBuffer, metrics::ImageGenerationOutput, model::data::CAState};

use super::{
    data::{CABatcher, CADataset},
    original_model::OriginalModelConfig,
};

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 1024)]
    pub pool_size: usize,

    #[config(default = 8000)]
    pub num_epochs: usize,
    #[config(default = 2e-3)]
    pub lr: f32,
    #[config(default = 0)]
    pub seed: u64,

    pub model: OriginalModelConfig,
    pub optimizer: AdamConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub type TrainingFeedback<B> = (
    Tensor<<B as AutodiffBackend>::InnerBackend, 4>,
    Tensor<<B as AutodiffBackend>::InnerBackend, 4>,
);

impl TrainingConfig {
    pub fn train<B: AutodiffBackend>(
        &self,
        artifact_dir: &str,
        image: &RgbaImageBuffer,
        device: B::Device,
        feedback_sender: Option<Sender<TrainingFeedback<B>>>,
    ) {
        let span = span!(Level::TRACE, "train", artifact_dir);
        let _enter = span.enter();

        let dataset = CADataset::new_from_image(
            image,
            crate::model::data::InitMethod::CenterPixel,
            self.pool_size,
        );

        create_artifact_dir(artifact_dir);
        self.save(format!("{}/self.json", artifact_dir))
            .expect("self should be saved successfully");

        B::seed(self.seed);
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut model = self.model.init::<B>(&device);
        let mut optim = self.optimizer.init();

        let batcher_train = CABatcher::<B>::new(device.clone());
        // validation phase does not require autodiff -> use inner backend
        let batcher_valid = CABatcher::<B::InnerBackend>::new(device.clone());

        // split into train and validation sets
        let (train_data, test_data) = dataset.split(0.8);

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(self.batch_size)
            .shuffle(self.seed)
            .num_workers(self.num_workers)
            .build(train_data);
        let dataloader_test = DataLoaderBuilder::new(batcher_valid)
            .batch_size(self.batch_size)
            .shuffle(self.seed)
            .num_workers(self.num_workers)
            .build(test_data);

        info!(
            "Training for {} epochs with seed: {}",
            self.num_epochs, self.seed
        );
        for _epoch in 1..self.num_epochs + 1 {
            for (_iteration, batch) in dataloader_train.iter().enumerate() {
                let steps = rng.random_range(64..=96);
                let output =
                    model.forward_regression(batch.initial_states, batch.expected_results, steps);
                let ImageGenerationOutput {
                    loss,
                    output,
                    targets,
                } = output;
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(self.lr.into(), model, grads);
                info!(
                    "Epoch: {}, Iteration: {}, Loss: {}",
                    _epoch,
                    _iteration,
                    loss.mean().into_scalar().elem::<f32>()
                );
                if let Some(sender) = feedback_sender.as_ref() {
                    if let Err(e) = sender.send((output.inner(), targets.inner())) {
                        warn!("Error sending feedback: {}", e);
                    }
                }
            }

            let model_valid = model.valid();

            for (_iteration, batch) in dataloader_test.iter().enumerate() {
                let steps = rng.random_range(64..=96);
                let output = model_valid.forward_regression(
                    batch.initial_states,
                    batch.expected_results,
                    steps,
                );
                let ImageGenerationOutput {
                    loss,
                    output,
                    targets,
                } = output;

                info!(
                    "Validation Epoch: {}, Iteration: {}, Loss: {}",
                    _epoch,
                    _iteration,
                    loss.mean().into_scalar().elem::<f32>()
                );
                if let Some(sender) = feedback_sender.as_ref() {
                    if let Err(e) = sender.send((output, targets)) {
                        warn!("Error sending feedback: {}", e);
                    }
                }
            }
        }
        todo!()
    }
}
