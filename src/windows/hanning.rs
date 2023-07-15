use core::{
    iter::{Enumerate, Map},
    ops::Mul,
};

#[allow(unused_imports)]
use micromath::F32Ext;

use crate::{Scalar, WindowFunction, PI};

pub struct Hanning;
impl<T> WindowFunction<T> for Hanning
where
    T: Copy + Mul<Scalar, Output = T>,
{
    type ItemMapper<'a, TIter : IntoIterator<Item = &'a T>> = Map<Enumerate<TIter::IntoIter>, fn((usize, &'a T)) -> T> where T : 'a;
    fn windowed<'a, const N: usize, TIter: IntoIterator<Item = &'a T>>(
        v: TIter,
    ) -> Self::ItemMapper<'a, TIter> {
        v.into_iter()
            .enumerate()
            .map(|(i, x)| *x * hanning(i as Scalar, N as Scalar))
    }
}

#[inline(always)]
fn hanning(i: Scalar, n: Scalar) -> Scalar {
    0.5 - 0.5 * (2.0 * PI * i / n).cos()
}
