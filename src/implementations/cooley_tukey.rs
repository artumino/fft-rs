use crate::{Allocator, Implementation, Scalar, PI};

use core::ops::{Add, Mul, Sub};

#[allow(unused_imports)]
use micromath::F32Ext;
use num_traits::FromPrimitive;

use super::CooleyTukey;

impl<T, const N: usize, A> Implementation<T, N, A> for CooleyTukey
where
    A: Allocator<T, N>,
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Scalar, Output = T>,
{
    fn fft(v: impl IntoIterator<Item = T>, spectrum: &mut A::Element) {
        let h = N >> 1;
        let mut swap_buffer = A::allocate();
        let (mut old, mut new) = (spectrum.as_mut(), swap_buffer.as_mut());
        let f_n: Scalar = FromPrimitive::from_usize(N).unwrap();

        for (i, x) in v.into_iter().enumerate() {
            old[i] = x;
        }

        let (mut sublen, mut stride) = (1, N);
        let mut swapped = false;
        while sublen < N {
            stride >>= 1;
            for i in 0..stride {
                for k in (0..N).step_by(stride << 1) {
                    let omega = (PI * (k as Scalar) / f_n).exp();
                    new[i + (k >> 1)] = old[i + k] + old[i + k + stride] * omega;
                    new[i + (k >> 1) + h] = old[i + k] - old[i + k + stride] * omega;
                }
            }
            (old, new) = (new, old);
            swapped = !swapped;
            sublen <<= 1;
        }

        if swapped {
            old.copy_from_slice(new);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{implementations::CooleyTukey, test::ComplexTestFixture};

    #[test]
    fn impulse_test() {
        ComplexTestFixture::<CooleyTukey>::impulse_test();
    }

    #[test]
    fn linearity_test() {
        ComplexTestFixture::<CooleyTukey>::linearity_test();
    }

    #[test]
    fn ground_truth_test() {
        ComplexTestFixture::<CooleyTukey>::ground_truth_test();
    }
}
