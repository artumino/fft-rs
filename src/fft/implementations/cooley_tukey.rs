use crate::fft::implementations::CooleyTukey;
use crate::fft::{Allocator, Implementation};
use core::ops::{Add, Mul, Sub};

#[allow(unused_imports)]
use micromath::F32Ext;

impl<T, const N: usize, A> Implementation<T, N, A> for CooleyTukey
where
    A: Allocator<T, N>,
    T: Copy + Mul<f32, Output = T> + Add<Output = T> + Sub<Output = T>,
{
    fn fft(v: &[T; N]) -> A::Element {
        let h = N >> 1;
        let (mut vec1, mut vec2) = (A::allocate(), A::allocate());
        let (mut old, mut new) = (vec1.as_mut(), vec2.as_mut());
        old.copy_from_slice(v);

        let (mut sublen, mut stride) = (1, N);
        let mut swapped = false;
        let mut omega;
        let f_n = N as f32;
        while sublen < N {
            stride >>= 1;
            for i in 0..stride {
                let mut k = 0;
                while k < N {
                    omega = (core::f32::consts::PI * (k as f32) / f_n).exp();
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
            return vec2;
        }

        vec1
    }
}
