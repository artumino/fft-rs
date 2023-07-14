use crate::implementations::CooleyTukey;
use crate::{Allocator, Implementation};

use core::ops::{Add, Mul, Sub};

#[allow(unused_imports)]
use micromath::F32Ext;
use num_traits::FromPrimitive;

impl<T, const N: usize, A> Implementation<T, N, A> for CooleyTukey
where
    A: Allocator<T, N>,
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + OmegaCalculator<T>
        + Mul<<T as OmegaCalculator<T>>::TMul, Output = T>,
{
    fn fft(v: impl IntoIterator<Item = T>, spectrum: &mut A::Element) {
        let h = N >> 1;
        let mut swap_buffer = A::allocate();
        let (mut old, mut new) = (spectrum.as_mut(), swap_buffer.as_mut());
        let f_n = FromPrimitive::from_usize(N).unwrap();

        for (i, x) in v.into_iter().enumerate() {
            old[i] = x;
        }

        let (mut sublen, mut stride) = (1, N);
        let mut swapped = false;
        while sublen < N {
            stride >>= 1;
            for i in 0..stride {
                let mut k = 0;
                while k < N {
                    let omega = T::calculate_omega(k, f_n);
                    new[i + (k >> 1)] = old[i + k] + old[i + k + stride] * omega;
                    new[i + (k >> 1) + h] = old[i + k] - old[i + k + stride] * omega;
                    k += 2 * stride;
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

pub trait OmegaCalculator<T>
where
    T: Copy + Mul<Self::TMul, Output = T>,
{
    type TMul: FromPrimitive + Copy;
    fn calculate_omega(k: usize, n: Self::TMul) -> Self::TMul;
}

impl<T> OmegaCalculator<T> for T
where
    T: Copy + Mul<f32, Output = T>,
{
    type TMul = f32;
    fn calculate_omega(k: usize, n: f32) -> f32 {
        (core::f32::consts::PI * (k as f32) / n).exp()
    }
}
