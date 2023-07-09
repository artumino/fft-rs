use core::{
    iter::{Enumerate, Map},
    ops::Mul,
    slice::Iter,
};

use crate::WindowFunction;

pub struct Hanning;
impl<T> WindowFunction<T> for Hanning
where
    T: Copy + Mul<f32, Output = T>,
{
    type ItemMapper<'a> = Map<Enumerate<Iter<'a, T>>, fn((usize, &'a T)) -> T> where T : 'a ;
    fn windowed<const N: usize>(v: &[T]) -> Self::ItemMapper<'_> {
        v.iter()
            .enumerate()
            .map(|(i, x)| (*x) * hanning(i as f32, N as f32))
    }
}

#[inline(always)]
fn hanning(i: f32, n: f32) -> f32 {
    0.5f32 - 0.5f32 * (2.0f32 * core::f32::consts::PI * i / n).cos()
}
