pub trait ToFfi {
    type T;
    fn to_ffi(&self) -> *mut Self::T;
}
