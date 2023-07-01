use crate::fft::Allocator;

pub struct ArrayAllocator;
impl<const N: usize> Allocator<f32, N> for ArrayAllocator {
    type Element = [f32; N];

    fn allocate() -> Self::Element {
        [0.0f32; N]
    }
}
