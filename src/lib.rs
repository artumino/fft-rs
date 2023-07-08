#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
#[macro_use]
extern crate alloc;

#[cfg(test)]
mod test;

use core::marker::PhantomData;

use self::implementations::CooleyTukey;

pub mod allocators;
pub mod implementations;

pub trait Allocator<T, const N: usize> {
    type Element: AsMut<[T]> + AsRef<[T]> + Sized;
    fn allocate() -> Self::Element;
}

pub trait Implementation<T, const N: usize, A>
where
    A: Allocator<T, N>,
{
    fn fft(v: &[T; N], spectrum: &mut A::Element);
}

pub struct Engine<T, const N: usize, I, A>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    T: Copy,
{
    impl_marker: PhantomData<I>,
    allocator_marker: PhantomData<A>,
    element_marker: PhantomData<T>,
}

type DefaultImpl = CooleyTukey;

#[cfg(feature = "alloc")]
type DefaultAllocator = allocators::boxed::BoxedAllocator;

#[cfg(not(feature = "alloc"))]
type DefaultAllocator = allocators::array::ArrayAllocator;

impl<const N: usize> Default for Engine<f32, N, DefaultImpl, DefaultAllocator> {
    fn default() -> Engine<f32, N, DefaultImpl, DefaultAllocator> {
        Engine {
            impl_marker: PhantomData,
            allocator_marker: PhantomData,
            element_marker: PhantomData,
        }
    }
}

impl<T, const N: usize, I, A> Engine<T, N, I, A>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    T: Copy,
{
    pub fn new() -> Engine<T, N, I, A> {
        Engine {
            impl_marker: PhantomData,
            allocator_marker: PhantomData,
            element_marker: PhantomData,
        }
    }

    pub fn fft(&self, v: &[T; N], spectrum: &mut <A as Allocator<T, N>>::Element) {
        <I as Implementation<T, N, A>>::fft(v, spectrum)
    }
}
