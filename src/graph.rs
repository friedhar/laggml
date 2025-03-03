use crate::{
    context::Context,
    sys::{ggml_mul, ggml_mul_mat},
    tensor::Tensor2d,
    to_ffi::ToFfi,
};

pub struct Graph {
    ctx: Context,
}

impl Graph {
    pub fn new(ctx: Context) -> Graph {
        Graph { ctx }
    }

    pub fn matmul(&mut self, a: &Tensor2d, b: &Tensor2d) -> u8 {
        let o = unsafe { ggml_mul_mat(self.ctx.to_ffi(), a.to_ffi(), b.to_ffi()) };
        0
    }
}
