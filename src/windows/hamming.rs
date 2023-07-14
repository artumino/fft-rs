use core::{
    iter::{Enumerate, Map},
    ops::Mul,
};

#[allow(unused_imports)]
use micromath::F32Ext;

use crate::WindowFunction;

pub struct Hamming;
impl<T> WindowFunction<T> for Hamming
where
    T: Copy + Mul<f32, Output = T>,
{
    type ItemMapper<'a, TIter: IntoIterator<Item = &'a T>> = Map<Enumerate<TIter::IntoIter>, fn((usize, &'a T)) -> T> where T : 'a;
    fn windowed<'a, const N: usize, TIter: IntoIterator<Item = &'a T>>(
        v: TIter,
    ) -> Self::ItemMapper<'a, TIter> {
        v.into_iter()
            .enumerate()
            .map(|(i, x)| *x * hamming(i as f32, N as f32))
    }
}

#[inline(always)]
fn hamming(i: f32, n: f32) -> f32 {
    0.54f32 - 0.46f32 * (2.0f32 * core::f32::consts::PI * i / n).cos()
}
