use crate::implementations::Naive;
use crate::{Allocator, ImgUnit, Implementation, Scalar, PI};

use core::ops::{Add, Mul, Sub};

#[allow(unused_imports)]
use micromath::F32Ext;

use crate::ComplexFloat;

impl<T, const N: usize, A> Implementation<T, N, A> for Naive
where
    A: Allocator<T, N>,
    T: Copy + Add<Output = T> + Sub<Output = T> + ImgUnit + Mul<Scalar, Output = T> + ComplexFloat,
{
    type Cache = ();
    type InternalBuffer = A::Element;
    fn fft(
        v: impl IntoIterator<Item = T>,
        spectrum: &mut A::Element,
        buffer: &mut Self::InternalBuffer,
        _cache: &Self::Cache,
    ) {
        let f_n = N as Scalar;
        let unit = T::img_unit();
        let buffer = buffer.as_mut();

        for (idx, x) in v.into_iter().enumerate() {
            buffer[idx] = x;
        }

        let spectrum = spectrum.as_mut();
        for (i, x) in spectrum.iter_mut().enumerate().take(N) {
            for (j, y) in buffer.iter().enumerate().take(N) {
                let omega = -(2.0 * PI * (i as Scalar) * (j as Scalar)) / f_n;
                *x = *x + *y * (unit * omega).exp();
            }
        }
    }

    fn init_cache() -> Self::Cache {
        Self::Cache::default()
    }

    fn init_buffer() -> Self::InternalBuffer {
        A::allocate()
    }
}

#[cfg(test)]
mod test {
    use crate::{implementations::Naive, test::ComplexTestFixture};

    #[test]
    fn impulse_test() {
        ComplexTestFixture::<Naive>::impulse_test();
    }

    #[test]
    fn linearity_test() {
        ComplexTestFixture::<Naive>::linearity_test();
    }

    #[test]
    fn ground_truth_test() {
        ComplexTestFixture::<Naive>::ground_truth_test();
    }
}
