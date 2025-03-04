use crate::{
    sys::{ggml_context, ggml_init, ggml_init_params},
    to_ffi::ToFfi,
};

pub struct Context {
    params: ggml_init_params,
    pub(crate) x: *mut ggml_context,
}

impl Context {
    pub fn new(capacity: usize, no_alloc: bool) -> Context {
        let params = ggml_init_params {
            mem_size: capacity,
            mem_buffer: std::ptr::null_mut(),
            no_alloc,
        };
        let x = unsafe { ggml_init(params) };
        Context { params, x }
    }
}

impl ToFfi for Context {
    type T = ggml_context;
    fn to_ffi(&self) -> *mut Self::T {
        self.x as *mut Self::T
    }
}
