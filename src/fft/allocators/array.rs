use crate::fft::Allocator;

pub struct ArrayAllocator;
impl<T, const N: usize> Allocator<T, N> for ArrayAllocator
where
    T: Default + Copy,
{
    type Element = [T; N];

    fn allocate() -> Self::Element {
        [T::default(); N]
    }
}
