#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
#[macro_use]
extern crate alloc;

#[cfg(test)]
mod test;

use core::marker::PhantomData;

use windows::Rect;

use self::implementations::CooleyTukey;

pub mod allocators;
pub mod implementations;
pub mod windows;

pub trait WindowFunction<T>
where
    T: Copy,
{
    type ItemMapper<'a>: ExactSizeIterator<Item = T>
    where
        T: 'a;
    fn windowed<const N: usize>(v: &[T]) -> Self::ItemMapper<'_>;
}

pub trait Allocator<T, const N: usize> {
    type Element: AsMut<[T]> + AsRef<[T]> + Sized;
    fn allocate() -> Self::Element;
}

pub trait Implementation<T, const N: usize, A>
where
    A: Allocator<T, N>,
    T: Copy,
{
    fn fft(v: impl ExactSizeIterator<Item = T>, spectrum: &mut A::Element);
}

pub struct Engine<T, const N: usize, I, W, A>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    W: WindowFunction<T>,
    T: Copy,
{
    impl_marker: PhantomData<I>,
    allocator_marker: PhantomData<A>,
    window_marker: PhantomData<W>,
    element_marker: PhantomData<T>,
}

#[cfg(feature = "alloc")]
type DefaultAllocator = allocators::boxed::BoxedAllocator;

#[cfg(not(feature = "alloc"))]
type DefaultAllocator = allocators::array::ArrayAllocator;

impl<const N: usize> Default for Engine<f32, N, CooleyTukey, Rect, DefaultAllocator> {
    fn default() -> Engine<f32, N, CooleyTukey, Rect, DefaultAllocator> {
        Engine {
            impl_marker: PhantomData,
            allocator_marker: PhantomData,
            window_marker: PhantomData,
            element_marker: PhantomData,
        }
    }
}

impl<T, const N: usize, I, W, A> Engine<T, N, I, W, A>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    W: WindowFunction<T>,
    T: Copy,
{
    pub fn new() -> Engine<T, N, I, W, A> {
        Engine {
            impl_marker: PhantomData,
            allocator_marker: PhantomData,
            window_marker: PhantomData,
            element_marker: PhantomData,
        }
    }

    pub fn fft(&self, v: &[T; N], spectrum: &mut <A as Allocator<T, N>>::Element) {
        <I as Implementation<T, N, A>>::fft(W::windowed::<N>(v), spectrum)
    }
}
