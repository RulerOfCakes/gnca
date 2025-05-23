use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Tensor, TensorData},
};

use crate::imageutils::RgbaImageBuffer;

#[derive(Clone, Debug)]
pub struct CAState(pub Vec<Vec<Vec<f32>>>); // height, width, visible channels

impl CAState {
    pub fn new_single_pixel(height: usize, width: usize, channels: usize) -> Self {
        let mut base_vec = vec![vec![vec![0.0; channels]; width]; height];
        base_vec[height / 2][width / 2] = vec![1.0; channels];
        // set RGB channels to 0
        base_vec[height / 2][width / 2][0] = 0.0;
        base_vec[height / 2][width / 2][1] = 0.0;
        base_vec[height / 2][width / 2][2] = 0.0;
        CAState(base_vec)
    }
    pub fn new_from_image(buffer: &RgbaImageBuffer) -> Self {
        let (height, width) = (buffer.height() as usize, buffer.width() as usize);
        let channels = 4; // RGBA
        let mut base_vec = vec![vec![vec![0.0; channels]; width]; height];
        for i in 0..height {
            for j in 0..width {
                let pixel = buffer.get_pixel(j as u32, i as u32);
                base_vec[i][j] = vec![
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                    pixel[3] as f32 / 255.0,
                ];
            }
        }
        CAState(base_vec)
    }
}

#[derive(Clone, Debug)]
pub struct CAData {
    pub state: CAState,
    pub expected: CAState,
}

pub enum InitMethod {
    CenterPixel,
    Gaussian,
}

impl CAData {
    pub fn new(expected: CAState, init_method: &InitMethod) -> Self {
        match init_method {
            InitMethod::CenterPixel => CAData {
                state: CAState::new_single_pixel(
                    expected.0.len(),
                    expected.0[0].len(),
                    expected.0[0][0].len(),
                ),
                expected,
            },
            InitMethod::Gaussian => unimplemented!(),
        }
    }
}

impl<B: Backend> From<Tensor<B, 3>> for CAState {
    fn from(value: Tensor<B, 3>) -> Self {
        let [height, width, channels] = value.dims();

        let flat_data = value.to_data();
        let mut iter: Box<dyn Iterator<Item = f32>> = flat_data.iter();

        let mut base_vec = vec![vec![vec![0.0; channels]; width]; height];
        for i in 0..height {
            for j in 0..width {
                for k in 0..channels {
                    base_vec[i][j][k] = iter.next().unwrap();
                }
            }
        }
        CAState(base_vec)
    }
}

#[derive(Clone, Debug)]
pub struct CADataset {
    pub data: Vec<CAData>,
}

impl CADataset {
    pub fn new(data: Vec<CAData>) -> Self {
        CADataset { data }
    }
    pub fn new_from_image(
        image: &RgbaImageBuffer,
        init_method: InitMethod,
        num_samples: usize,
    ) -> Self {
        let data = vec![CAData::new(CAState::new_from_image(image), &init_method); num_samples];
        CADataset { data }
    }
    pub fn split(self, ratio: f32) -> (Self, Self) {
        let split_index = (self.data.len() as f32 * ratio).round() as usize;
        let (train_data, valid_data) = self.data.split_at(split_index);
        (
            CADataset::new(train_data.to_vec()),
            CADataset::new(valid_data.to_vec()),
        )
    }
}

impl Dataset<CAData> for CADataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<CAData> {
        if index < self.data.len() {
            Some(self.data[index].clone())
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct CABatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> CABatcher<B> {
    pub fn new(device: B::Device) -> Self {
        CABatcher { device }
    }
}

// Burn expects the batch to be in `NCHW` format.
#[derive(Clone, Debug)]
pub struct CABatch<B: Backend> {
    pub initial_states: Tensor<B, 4>,
    pub expected_results: Tensor<B, 4>,
}

impl<B: Backend> Batcher<CAData, CABatch<B>> for CABatcher<B> {
    // The input data must consist of 2D images with **the same number of channels**.
    fn batch(&self, data: Vec<CAData>) -> CABatch<B> {
        let initial_states = data
            .iter()
            .map(|cadata| {
                let (height, width) = (cadata.state.0.len(), cadata.state.0[0].len());
                let channels = cadata.state.0[0][0].len();
                let flattened_state = cadata
                    .state
                    .0
                    .clone()
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>();
                TensorData::new(flattened_state, [height, width, channels])
            })
            .map(|tdata| Tensor::<B, 3>::from_data(tdata, &self.device))
            .map(|tensor| tensor.unsqueeze_dim(0))
            .collect();
        let expected_results = data
            .iter()
            .map(|cadata| {
                let (height, width) = (cadata.expected.0.len(), cadata.expected.0[0].len());
                let channels = cadata.expected.0[0][0].len();
                let flattened_expected = cadata
                    .expected
                    .0
                    .clone()
                    .into_iter()
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>();
                TensorData::new(flattened_expected, [height, width, channels])
            })
            .map(|tdata| Tensor::<B, 3>::from_data(tdata, &self.device))
            .map(|tensor| tensor.unsqueeze_dim(0))
            .collect();

        // N H W C
        let initial_states = Tensor::cat(initial_states, 0).to_device(&self.device);
        let expected_results = Tensor::cat(expected_results, 0).to_device(&self.device);

        // convert to N C H W
        let initial_states = initial_states.permute([0, 3, 1, 2]);
        let expected_results = expected_results.permute([0, 3, 1, 2]);
        CABatch {
            initial_states,
            expected_results,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{TestBackend, TestDevice};
    use burn::tensor::Shape;

    #[test]
    fn test_castate() {
        let device = TestDevice::default();
        let data = CAState(vec![vec![vec![1., 3., 2., 4.]; 5]; 3]);
        let tensor_data = TensorData::new(
            data.0
                .clone()
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>(),
            [3, 5, 4],
        );
        let tensor: Tensor<TestBackend, 3> = Tensor::from_data(tensor_data, &device);
        let casted: CAState = tensor.into();
        assert_eq!(data.0, casted.0);
    }

    #[test]
    fn test_batcher() {
        let device = TestDevice::default();
        let batcher: CABatcher<TestBackend> = CABatcher::new(device);
        let data = vec![
            CAData {
                state: CAState(vec![vec![vec![0.0; 4]; 5]; 3]),
                expected: CAState(vec![vec![vec![1.0; 4]; 5]; 3]),
            },
            CAData {
                state: CAState(vec![vec![vec![2.0; 4]; 5]; 3]),
                expected: CAState(vec![vec![vec![3.0; 4]; 5]; 3]),
            },
        ]; // 2D images with RGBA channels
        let batch = batcher.batch(data);

        // Output dimension should be in `NCHW` format
        assert_eq!(batch.initial_states.shape(), Shape::new([2, 4, 3, 5]));
        assert_eq!(batch.expected_results.shape(), Shape::new([2, 4, 3, 5]));

        let binding = batch.expected_results.to_data();
        let expected_batch: &[f32] = binding.as_slice().unwrap();
        for i in 0..2 {
            for j in 0..4 {
                for k in 0..3 {
                    for l in 0..5 {
                        assert_eq!(
                            expected_batch[i * 4 * 3 * 5 + j * 3 * 5 + k * 5 + l],
                            ((i * 2) + 1) as f32
                        );
                    }
                }
            }
        }

        let data = vec![CAData {
            state: CAState(vec![vec![vec![0.0; 4]; 5]; 3]),
            expected: CAState(vec![vec![vec![0., 1., 2., 3.]; 5]; 3]),
        }];
        let batch = batcher.batch(data);

        let binding = batch.expected_results.to_data();
        let initial_batch: &[f32] = binding.as_slice().unwrap();
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..5 {
                    assert_eq!(initial_batch[i * 3 * 5 + j * 5 + k], i as f32);
                }
            }
        }
    }
}
