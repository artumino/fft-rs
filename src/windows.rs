use core::{iter::Copied, ops::Mul};

use crate::{Scalar, WindowFunction};

pub mod hamming;
pub mod hanning;

pub struct Rect;
impl<T> WindowFunction<T> for Rect
where
    T: Copy + Mul<Scalar, Output = T>,
{
    type ItemMapper<'a, TIter: IntoIterator<Item = &'a T>> = Copied<TIter::IntoIter> where T : 'a;
    fn windowed<'a, const N: usize, TIter: IntoIterator<Item = &'a T>>(
        v: TIter,
    ) -> Self::ItemMapper<'a, TIter>
    where
        T: 'a,
    {
        v.into_iter().copied()
    }
}
