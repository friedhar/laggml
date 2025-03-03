use std::ffi::c_void;

use crate::{
    context::Contect,
    data_type::DataType,
    sys::{ggml_nbytes, ggml_new_tensor_2d, ggml_tensor, ggml_type_GGML_TYPE_F32, GGML_FILE_MAGIC},
    to_ffi::ToFfi,
};

pub struct Tensor2d {
    x: *mut ggml_tensor,
}

impl Tensor2d {
    pub fn new<T: DataType>(ctx: Contect, data: &[T::InitType], cols: i64, rows: i64) -> Tensor2d {
        let x = unsafe { ggml_new_tensor_2d(ctx.x, T::ggml_type(), cols, rows) };
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const c_void,
                (*x).data,
                ggml_nbytes(x),
            );
        }
        Tensor2d { x }
    }
}

impl ToFfi for Tensor2d {
    type T = ggml_tensor;
    fn to_ffi(&self) -> *mut Self::T {
        self.x as *mut Self::T
    }
}
