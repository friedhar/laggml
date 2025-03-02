use crate::sys::ggml_init_params;

pub struct GgmlContext {
    params: ggml_init_params,
}

impl GgmlContext {
    pub fn new(capacity: usize, no_alloc: bool) -> GgmlContext {
        GgmlContext {
            params: ggml_init_params {
                mem_size: capacity,
                mem_buffer: std::ptr::null_mut(),
                no_alloc,
            },
        }
    }
}
