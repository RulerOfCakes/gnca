use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Tensor, TensorData},
};

#[derive(Clone, Debug)]
pub struct CAData {
    pub state: Vec<Vec<Vec<f32>>>,    // height, width, visible channels
    pub expected: Vec<Vec<Vec<f32>>>, // height, width, visible channels
}

#[derive(Clone, Debug)]
pub struct CADataset {
    pub data: Vec<CAData>,
}

impl CADataset {
    pub fn new(data: Vec<CAData>) -> Self {
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
                let (height, width) = (cadata.state.len(), cadata.state[0].len());
                let channels = cadata.state[0][0].len();
                let flattened_state = cadata
                    .state
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
                let (height, width) = (cadata.expected.len(), cadata.expected[0].len());
                let channels = cadata.expected[0][0].len();
                let flattened_expected = cadata
                    .expected
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
    fn test_batcher() {
        let device = TestDevice::default();
        let batcher: CABatcher<TestBackend> = CABatcher::new(device);
        let data = vec![
            CAData {
                state: vec![vec![vec![0.0; 4]; 5]; 3],
                expected: vec![vec![vec![1.0; 4]; 5]; 3],
            },
            CAData {
                state: vec![vec![vec![2.0; 4]; 5]; 3],
                expected: vec![vec![vec![3.0; 4]; 5]; 3],
            },
        ]; // 2D images with RGBA channels
        let batch = batcher.batch(data);

        // Output dimension should be in `NHWC` format
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
            state: vec![vec![vec![0.0; 4]; 5]; 3],
            expected: vec![vec![vec![0., 1., 2., 3.]; 5]; 3],
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
