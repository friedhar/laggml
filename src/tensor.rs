use crate::{
    context::GgmlContext,
    sys::{ggml_new_tensor_2d, ggml_tensor, ggml_type_GGML_TYPE_F32, GGML_FILE_MAGIC},
};

pub struct Tensor2d {
    x: *mut ggml_tensor,
}

pub enum DataType {
    F32,
}

impl DataType {
    pub fn to_ggml(&self) -> u32 {
        use DataType::*;
        match self {
            F32 => ggml_type_GGML_TYPE_F32,
        }
    }
}

impl Tensor2d {
    pub fn new(ctx: GgmlContext, data_type: DataType, cols: i64, rows: i64) -> Tensor2d {
        let x = unsafe { ggml_new_tensor_2d(ctx.x, data_type.to_ggml(), cols, rows) };
        Tensor2d { x }
    }
}
