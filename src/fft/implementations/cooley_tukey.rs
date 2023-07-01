use crate::fft::implementations::CooleyTukey;
use crate::fft::{Allocator, Implementation};
use core::f32::consts::PI;

#[allow(unused_imports)]
use micromath::F32Ext;

impl<const N: usize, A> Implementation<f32, N, A> for CooleyTukey
where
    A: Allocator<f32, N>,
{
    fn fft(v: &[f32; N]) -> A::Element {
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
                    omega = (PI * (k as f32) / f_n).exp();
                    new[i + (k >> 1)] = old[i + k] + omega * old[i + k + stride];
                    new[i + (k >> 1) + h] = old[i + k] - omega * old[i + k + stride];
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