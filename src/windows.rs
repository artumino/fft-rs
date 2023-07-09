use core::{iter::Copied, ops::Mul, slice::Iter};

use crate::WindowFunction;

pub mod hamming;
pub mod hanning;

pub struct Rect;
impl<T> WindowFunction<T> for Rect
where
    T: Copy + Mul<f32, Output = T>,
{
    type ItemMapper<'a> = Copied<Iter<'a, T>> where T : 'a ;
    fn windowed<const N: usize>(v: &[T]) -> Self::ItemMapper<'_> {
        v.iter().copied()
    }
}
