use burn::{
    prelude::Backend,
    tensor::{Tensor, Transaction},
    train::metric::{Adaptor, ItemLazy, LossInput},
};
use burn_ndarray::NdArray;

pub struct ImageGenerationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Adaptor<LossInput<B>> for ImageGenerationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> ItemLazy for ImageGenerationOutput<B> {
    type ItemSync = ImageGenerationOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");
        let device = &Default::default();

        ImageGenerationOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> ImageGenerationOutput<B> {
    pub fn new(loss: Tensor<B, 1>, output: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        ImageGenerationOutput {
            loss,
            output,
            targets,
        }
    }
}
