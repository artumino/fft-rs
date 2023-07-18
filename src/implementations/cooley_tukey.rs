use crate::{Allocator, ImgUnit, Implementation, Scalar, PI};

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use core::ops::{Add, Mul, Sub};

#[allow(unused_imports)]
use micromath::F32Ext;

use crate::ComplexFloat;

use super::CooleyTukey;

#[cfg(feature = "alloc")]
pub struct TwiddleCache<T, const N: usize> {
    cache: Box<[T]>,
}

#[cfg(feature = "alloc")]
impl<T, const N: usize> TwiddleCache<T, N>
where
    T: Copy + ImgUnit + ComplexFloat + Mul<Scalar, Output = T>,
{
    pub fn get(&self, i: usize) -> T {
        self.cache[i]
    }
}

#[cfg(feature = "alloc")]
impl<T, const N: usize> Default for TwiddleCache<T, N>
where
    T: Copy + ImgUnit + ComplexFloat + Mul<Scalar, Output = T>,
{
    fn default() -> Self {
        let half_n = N >> 1;
        let f_n = N as Scalar;
        let mut twiddles = vec![T::zero(); half_n].into_boxed_slice();
        for (i, twiddle) in twiddles.iter_mut().enumerate() {
            *twiddle = calculate_twiddle::<T>(i, f_n);
        }
        Self { cache: twiddles }
    }
}

#[inline]
fn bit_reversal(k: usize, log_n: usize) -> usize {
    let mut k = k;
    let mut r = 0;
    for _ in 0..log_n {
        r <<= 1;
        r |= k & 1;
        k >>= 1;
    }
    r
}

fn calculate_twiddle<T>(i: usize, f_n: Scalar) -> T
where
    T: Copy + ImgUnit + ComplexFloat + Mul<Scalar, Output = T>,
{
    let omega = -2.0 * PI * (i as Scalar) / f_n;
    (T::img_unit() * omega).exp()
}

impl<T, const N: usize, A> Implementation<T, N, A> for CooleyTukey
where
    A: Allocator<T, N>,
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Scalar, Output = T> + ImgUnit + ComplexFloat,
{
    #[cfg(feature = "alloc")]
    type Cache = TwiddleCache<T, N>;

    #[cfg(not(feature = "alloc"))]
    type Cache = ();

    fn fft(v: impl IntoIterator<Item = T>, spectrum: &mut A::Element, cache: &Self::Cache) {
        // Since we could be in a circular buffer, we copy data to a local buffer.
        // TODO: Maybe avoid this (without complicating the interface too much)
        let mut buffer = A::allocate();
        let buffer = buffer.as_mut();
        let spectrum = spectrum.as_mut();

        for (i, x) in v.into_iter().enumerate() {
            buffer[i] = x;
        }

        // 1. Bit-reversal permutation
        let log_n = N.trailing_zeros() as usize;
        let half_n = N >> 1;

        #[cfg(not(feature = "alloc"))]
        let f_n = N as Scalar;

        for i in (0..N).step_by(2) {
            let j = bit_reversal(i, log_n);
            let k = bit_reversal(i + 1, log_n);

            spectrum[i] = buffer[j] + buffer[k];
            spectrum[i + 1] = buffer[j] - buffer[k];
        }

        // 2. Butterfly computation
        let mut sublen = half_n;
        let mut stride = 2;
        for _ in 1..log_n {
            sublen >>= 1;
            for j in (0..N).step_by(stride * 2) {
                #[cfg(feature = "alloc")]
                let mut m = 0;

                for k in j..j + stride {
                    #[cfg(feature = "alloc")]
                    let twiddle = cache.get(m);

                    #[cfg(not(feature = "alloc"))]
                    let twiddle = calculate_twiddle::<T>(k, f_n);

                    let a = spectrum[k + stride] * twiddle;
                    let b = spectrum[k];
                    spectrum[k] = b + a;
                    spectrum[k + stride] = b - a;

                    #[cfg(feature = "alloc")]
                    {
                        m += sublen;
                    }
                }
            }
            stride <<= 1;
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
