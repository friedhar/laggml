use crate::sys::{ggml_context, ggml_init, ggml_init_params};

pub struct GgmlContext {
    params: ggml_init_params,
    x: *mut ggml_context,
}

impl GgmlContext {
    pub fn new(capacity: usize, no_alloc: bool) -> GgmlContext {
        let params = ggml_init_params {
            mem_size: capacity,
            mem_buffer: std::ptr::null_mut(),
            no_alloc,
        };
        let x = unsafe { ggml_init(params) };
        GgmlContext { params, x }
    }
}
