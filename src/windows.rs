use core::{iter::Copied, ops::Mul};

use crate::{Scalar, WindowFunction};

pub mod hamming;
pub mod hanning;

#[derive(Default)]
pub struct Rect;

impl<T, const N: usize> WindowFunction<T, N> for Rect
where
    T: Copy + Mul<Scalar, Output = T>,
{
    type ItemMapper<'a, TIter: IntoIterator<Item = &'a T>> = Copied<TIter::IntoIter> where T : 'a;
    fn windowed<'a, TIter: IntoIterator<Item = &'a T>>(
        &self,
        v: TIter,
    ) -> Self::ItemMapper<'a, TIter>
    where
        T: 'a,
    {
        v.into_iter().copied()
    }
}
