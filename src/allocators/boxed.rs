use crate::Allocator;
use alloc::boxed::Box;

pub struct BoxedAllocator;
impl<T, const N: usize> Allocator<T, N> for BoxedAllocator
where
    T: Default + Copy,
{
    type Element = Box<[T]>;

    fn allocate() -> Self::Element {
        vec![T::default(); N].into_boxed_slice()
    }
}
