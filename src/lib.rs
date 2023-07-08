#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
#[macro_use]
extern crate alloc;

#[cfg(test)]
mod test;

use core::{marker::PhantomData, ops::Mul};

use num_traits::FromPrimitive;
use windows::Rect;

use self::implementations::CooleyTukey;

pub mod allocators;
pub mod implementations;
pub mod windows;

pub trait WindowFunction<T>
where
    T: Copy + Mul<Self::TMul, Output = T>,
{
    type TMul: FromPrimitive + Copy;
    fn calculate(i: Self::TMul, n: Self::TMul) -> Self::TMul;
}

pub trait Allocator<T, const N: usize> {
    type Element: AsMut<[T]> + AsRef<[T]> + Sized;
    fn allocate() -> Self::Element;
}

pub trait Implementation<T, const N: usize, W, A>
where
    A: Allocator<T, N>,
    W: WindowFunction<T>,
    T: Copy + Mul<W::TMul, Output = T>,
{
    fn fft(v: &[T; N], spectrum: &mut A::Element);
}

pub struct Engine<T, const N: usize, I, W, A>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, W, A>,
    W: WindowFunction<T>,
    T: Copy + Mul<W::TMul, Output = T>,
{
    impl_marker: PhantomData<I>,
    allocator_marker: PhantomData<A>,
    window_marker: PhantomData<W>,
    element_marker: PhantomData<T>,
}

type DefaultImpl = CooleyTukey;
type DefaultWindowingFunction = Rect;

#[cfg(feature = "alloc")]
type DefaultAllocator = allocators::boxed::BoxedAllocator;

#[cfg(not(feature = "alloc"))]
type DefaultAllocator = allocators::array::ArrayAllocator;

impl<const N: usize> Default for Engine<f32, N, DefaultImpl, DefaultWindowingFunction, DefaultAllocator> {
    fn default() -> Engine<f32, N, DefaultImpl, DefaultWindowingFunction, DefaultAllocator> {
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
    I: Implementation<T, N, W, A>,
    W: WindowFunction<T>,
    T: Copy + Mul<W::TMul, Output = T>,
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
        <I as Implementation<T, N, W, A>>::fft(v, spectrum)
    }
}
