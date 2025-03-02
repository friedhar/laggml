use crate::{
    context::GgmlContext,
    sys::{ggml_new_tensor, ggml_tensor},
};

pub struct Tensor {
    inner: ggml_tensor,
}

impl Tensor {
    pub fn new(ctx: GgmlContext) -> Tensor {
        Tensor { inner: x }
    }
}
