pub mod context;
pub mod data_type;
pub mod graph;
pub mod sys;
pub mod tensor;

#[cfg(test)]
mod tests {
    use crate::sys::{ggml_context, ggml_init, ggml_new_tensor, ggml_tensor};

    #[test]
    fn test0() {}
}
