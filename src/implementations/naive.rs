use crate::implementations::Naive;
use crate::{Allocator, Implementation, Scalar};

use core::ops::{Add, Mul, Sub};

#[allow(unused_imports)]
use micromath::F32Ext;
use num_complex::{Complex, ComplexFloat};
use num_traits::{FromPrimitive, One, Zero};

impl<T, const N: usize, A> Implementation<T, N, A> for Naive
where
    A: Allocator<T, N>,
    T: Copy + Add<Output = T> + Sub<Output = T> + ImgUnit + Mul<Scalar, Output = T> + ComplexFloat,
{
    fn fft(v: impl IntoIterator<Item = T>, spectrum: &mut A::Element) {
        let n_f: Scalar = FromPrimitive::from_usize(N).unwrap();
        let mut buffer = A::allocate();
        let unit = T::img_unit();
        let buffer = buffer.as_mut();

        for (idx, x) in v.into_iter().enumerate() {
            buffer[idx] = x;
        }

        let spectrum = spectrum.as_mut();
        for (i, x) in spectrum.iter_mut().enumerate().take(N) {
            for (j, y) in buffer.iter().enumerate().take(N) {
                let omega = (-1.0 * 2.0 * (i as Scalar) * (j as Scalar)) / n_f;
                *x = *x + *y * (unit * omega).exp();
            }
        }
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
