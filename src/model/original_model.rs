use burn::{
    nn::{
        Linear, LinearConfig, Relu,
        loss::{HuberLossConfig, Reduction},
    },
    prelude::*,
    tensor::{
        Distribution,
        module::{conv2d, max_pool2d},
        ops::ConvOptions,
    },
};
use serde::{Deserialize, Serialize};

use crate::metrics::ImageGenerationOutput;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum ModelType {
    Growing,
    Persistent,
    Regenerating,
}

#[derive(Config, Debug)]
pub(crate) struct OriginalModelConfig {
    pub input_dim: usize,

    #[config(default = "16")]
    pub state_channels: usize,
    #[config(default = "16")]
    pub target_padding: usize,
    #[config(default = "40")]
    pub target_size: usize,
    #[config(default = "1024")]
    pub pool_size: usize,
    #[config(default = "0.5")]
    pub cell_fire_rate: f64, // rate of cells chosen for stochastic update.

    #[config(default = "0.1")]
    pub alive_threshold: f64,

    pub model_type: ModelType,
}

#[derive(Module, Debug)]
pub(crate) struct OriginalModel<B: Backend> {
    dense128: Linear<B>,
    dense_out: Linear<B>,
    relu: Relu,

    state_channels: usize,
    alive_threshold: f32,
    cell_fire_rate: f32,
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

        OriginalModel {
            dense128: LinearConfig::new(self.input_dim, 128).init(device),
            dense_out: LinearConfig::new(128, self.state_channels).init(device),
            relu: Relu::new(),

            state_channels: self.state_channels,
            alive_threshold: self.alive_threshold as f32,
            cell_fire_rate: self.cell_fire_rate as f32,
        }
    }
}

const SOBEL_FILTER_X: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
const SOBEL_FILTER_Y: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

impl<B: Backend> OriginalModel<B> {
    fn perceive(&self, states: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = states.device();
        let conv_options: ConvOptions<2> = ConvOptions::new([1, 1], [1, 1], [1, 1], 1); // stride, padding, dilation, groups
        let sobel_weights_x = Tensor::from_floats(SOBEL_FILTER_X, &device);
        let sobel_weights_y = Tensor::from_floats(SOBEL_FILTER_Y, &device);
        // partial derivative estimation with sobel filters
        let grad_x = conv2d(states.clone(), sobel_weights_x, None, conv_options.clone());
        let grad_y = conv2d(states.clone(), sobel_weights_y, None, conv_options);

        Tensor::cat(vec![states, grad_x, grad_y], 1)
    }
    fn update(&self, perception: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.dense128.forward(perception);
        let x = self.relu.forward(x);
        self.dense_out.forward(x)
    }
    fn alive_masking(&self, states: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = states.device();
        let alpha_channels = states.clone().select(1, Tensor::from_ints([3], &device));
        let adjacent_max_alpha = max_pool2d(alpha_channels, [3, 3], [1, 1], [1, 1], [1, 1]);

        let alive_mask = adjacent_max_alpha
            .greater_elem(self.alive_threshold)
            .expand(states.clone().dims());

        Tensor::zeros(states.dims(), &device).mask_where(alive_mask, states)
    }
    pub fn forward(&self, states: Tensor<B, 4>, fire_rate: Option<f32>) -> Tensor<B, 4> {
        let device = states.device();
        let live_states = self.alive_masking(states);

        let perception = self.perceive(live_states.clone());

        let updated_states = self.update(perception);

        let dist = Distribution::Uniform(0., 1.);
        let stochastic_mask: Tensor<B, 4> = Tensor::random(updated_states.dims(), dist, &device);
        // A custom fire rate may be given for training / inference
        let stochastic_mask =
            stochastic_mask.greater_elem(fire_rate.unwrap_or(self.cell_fire_rate));

        live_states.mask_where(stochastic_mask, updated_states)
    }
    pub fn forward_regression(
        &self,
        states: Tensor<B, 4>,
        targets: Tensor<B, 4>,
        steps: usize,
    ) -> ImageGenerationOutput<B> {
        // first, expand the channels of the input states
        let mut shape = states.dims();
        let original_channels = shape[1];
        shape[1] = self.state_channels;
        let mut states = states.expand(shape);

        for _ in 0..steps {
            states = self.forward(states, None);
        }

        // reduce the channels again to match the target
        let output = states.slice([0..shape[0], 0..original_channels, 0..shape[2], 0..shape[3]]);
        let reduction = Reduction::Auto;
        let loss =
            HuberLossConfig::new(0.1)
                .init()
                .forward(output.clone(), targets.clone(), reduction);

        ImageGenerationOutput::new(loss, output, targets)
    }
}
