use burn::{config::Config, data::dataloader::DataLoaderBuilder, tensor::backend::AutodiffBackend};

use super::{
    data::{CABatcher, CADataset},
    original_model::OriginalModelConfig,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: OriginalModelConfig,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    pub seed: u64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    dataset: CADataset,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{}/config.json", artifact_dir))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = CABatcher::<B>::new(device.clone());
    // validation phase does not require autodiff -> use inner backend
    let batcher_valid = CABatcher::<B::InnerBackend>::new(device.clone());

    // split into train and validation sets
    let (train_data, test_data) = dataset.split(0.8);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_data);
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_data);

    // Custom training loop for BPTT
    todo!()
}
