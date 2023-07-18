#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
#[macro_use]
extern crate alloc;

#[cfg(test)]
pub(crate) mod test;

use core::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use num_complex::{Complex, ComplexFloat};
use num_traits::{One, Zero};
use windows::Rect;

use self::implementations::CooleyTukey;

pub mod allocators;
pub mod implementations;
pub mod windows;

pub trait WindowFunction<T>
where
    T: Copy,
{
    type ItemMapper<'a, TIter: IntoIterator<Item = &'a T>>: IntoIterator<Item = T>
    where
        T: 'a;
    fn windowed<'a, const N: usize, TIter: IntoIterator<Item = &'a T>>(
        v: TIter,
    ) -> Self::ItemMapper<'a, TIter>
    where
        T: 'a;
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
    type Cache: Default;
    fn fft(v: impl IntoIterator<Item = T>, spectrum: &mut A::Element, cache: &Self::Cache);
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
    cache: <I as Implementation<T, N, A>>::Cache,
}

#[cfg(feature = "alloc")]
type DefaultAllocator = allocators::boxed::BoxedAllocator;

#[cfg(not(feature = "alloc"))]
type DefaultAllocator = allocators::array::ArrayAllocator;

impl<T, const N: usize> Default for Engine<T, N, CooleyTukey, Rect, DefaultAllocator>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Scalar, Output = T>
        + ImgUnit
        + ComplexFloat
        + Default,
{
    fn default() -> Engine<T, N, CooleyTukey, Rect, DefaultAllocator> {
        Engine {
            impl_marker: PhantomData,
            allocator_marker: PhantomData,
            window_marker: PhantomData,
            element_marker: PhantomData,
            cache: <CooleyTukey as Implementation<T, N, DefaultAllocator>>::Cache::default(),
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
            cache: <I as Implementation<T, N, A>>::Cache::default(),
        }
    }

    pub fn fft<'a, TIter: IntoIterator<Item = &'a T>>(
        &self,
        v: TIter,
        spectrum: &mut <A as Allocator<T, N>>::Element,
    ) where
        T: 'a,
    {
        <I as Implementation<T, N, A>>::fft(W::windowed::<N, TIter>(v), spectrum, &self.cache);
    }
}

pub trait ImgUnit {
    fn img_unit() -> Self;
}

impl<T> ImgUnit for Complex<T>
where
    T: Zero + One,
{
    fn img_unit() -> Self {
        Complex::new(T::zero(), T::one())
    }
}

#[cfg(not(feature = "precision"))]
pub type Scalar = f32;

#[cfg(feature = "precision")]
pub type Scalar = f64;

#[cfg(not(feature = "precision"))]
pub const PI: Scalar = core::f32::consts::PI;

#[cfg(feature = "precision")]
pub const PI: Scalar = core::f64::consts::PI;
