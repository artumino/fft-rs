use crate::fft::Allocator;
use alloc::boxed::Box;

pub struct BoxedAllocator;
impl<const N: usize> Allocator<f32, N> for BoxedAllocator {
    type Element = Box<[f32]>;

    fn allocate() -> Self::Element {
        vec![0.0f32; N].into_boxed_slice()
    }
}
