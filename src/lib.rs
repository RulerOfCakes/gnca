pub mod imageutils;
pub mod metrics;
pub mod model;
pub mod ui;
#[cfg(test)]
mod test_utils {
    pub(crate) type TestBackend = burn::backend::Wgpu<f32, i32>;
    pub(crate) type TestDevice = burn::backend::wgpu::WgpuDevice;
}
