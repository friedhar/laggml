use crate::sys::ggml_type_GGML_TYPE_F32;

pub trait DataType {
    type InitType;
    fn ggml_type() -> u32;
}

pub struct F32 {}

impl DataType for F32 {
    type InitType = f32;
    fn ggml_type() -> u32 {
        ggml_type_GGML_TYPE_F32
    }
}
