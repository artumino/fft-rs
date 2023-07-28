use core::{
    iter::{Enumerate, Map},
    ops::Mul, cell::OnceCell,
};

#[allow(unused_imports)]
use micromath::F32Ext;

use crate::{Scalar, WindowFunction, PI};

#[derive(Default)]
pub struct Hamming<T>
{
    pub cached: OnceCell<[T; 2]>,
}

impl<T, const N: usize> WindowFunction<T, N> for Hamming<T>
where
    T: Copy + Mul<Scalar, Output = T>,
{
    type ItemMapper<'a, TIter: IntoIterator<Item = &'a T>> = Map<Enumerate<TIter::IntoIter>, fn((usize, &'a T)) -> T> where T : 'a;
    fn windowed<'a, TIter: IntoIterator<Item = &'a T>>(
        &self,
        v: TIter,
    ) -> Self::ItemMapper<'a, TIter> {
        v.into_iter()
            .enumerate()
            .map(|(i, x)| *x * hamming(i as Scalar, N as Scalar))
    }
}

#[inline(always)]
fn hamming(i: Scalar, n: Scalar) -> Scalar {
    0.54 - 0.46 * (2.0 * PI * i / n).cos()
}
