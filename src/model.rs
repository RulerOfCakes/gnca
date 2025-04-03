use burn::{
    nn::{Linear, Relu},
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum ModelType {
    Growing,
    Persistent,
    Regenerating,
}

#[derive(Config, Debug)]
pub(crate) struct OriginalModelConfig {
    pub input_dim: usize,
    pub output_dim: usize,

    #[config(default = "16")]
    pub state_channels: usize,
    #[config(default = "16")]
    pub target_padding: usize,
    #[config(default = "40")]
    pub target_size: usize,
    #[config(default = "8")]
    pub batch_size: usize,
    #[config(default = "1024")]
    pub pool_size: usize,
    #[config(default = "0.5")]
    pub cell_fire_rate: f64,

    pub model_type: ModelType,
}

#[derive(Module, Debug)]
pub(crate) struct OriginalModel<B: Backend> {
    dense128: Linear<B>,
    dense16: Linear<B>,
    relu: Relu,
}

impl OriginalModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> OriginalModel<B> {
        let use_pattern_pool = match self.model_type {
            ModelType::Growing => false,
            ModelType::Persistent => true,
            ModelType::Regenerating => true,
        };
        let damage_n = match self.model_type {
            ModelType::Growing => 0,
            ModelType::Persistent => 0,
            ModelType::Regenerating => 3,
        }; // number of patterns to damage in a batch
        todo!()
    }
}

impl<B: Backend> OriginalModel<B> {
    // input and output are both in batch tensors
    // the output must also be an image tensor
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 3> {
        todo!()
    }
}
