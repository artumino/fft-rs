use core::{
    iter::{Enumerate, Map},
    ops::Mul,
    slice::Iter,
};

#[allow(unused_imports)]
use micromath::F32Ext;

use crate::WindowFunction;

pub struct Hamming;
impl<T> WindowFunction<T> for Hamming
where
    T: Copy + Mul<f32, Output = T>,
{
    type ItemMapper<'a> = Map<Enumerate<Iter<'a, T>>, fn((usize, &'a T)) -> T> where T : 'a ;
    fn windowed<const N: usize>(v: &[T]) -> Self::ItemMapper<'_> {
        v.iter()
            .enumerate()
            .map(|(i, x)| (*x) * hamming(i as f32, N as f32))
    }
}

#[inline(always)]
fn hamming(i: f32, n: f32) -> f32 {
    0.54f32 - 0.46f32 * (2.0f32 * core::f32::consts::PI * i / n).cos()
}
