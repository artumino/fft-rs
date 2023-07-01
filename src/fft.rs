#[allow(unused_imports)]
use micromath::F32Ext;
#[allow(clippy::approx_constant)]
#[allow(clippy::excessive_precision)]
pub const PI: f32 = 3.141592653589793f32;

pub fn fft<const N: usize>(v: &[f32; N]) -> [f32; N] {
    let h = N >> 1;
    let (mut cloned_v, mut zero_v) = (*v, [0.0f32; N]);
    let (mut old, mut new) = (&mut cloned_v, &mut zero_v);
    let (mut sublen, mut stride) = (1, N);
    let mut swapped = false;
    let mut omega;
    let f_n = N as f32;
    while sublen < N {
        stride >>= 1;
        for i in 0..stride {
            for k in (0..N).step_by(2 * stride) {
                omega = (PI * (k as f32) / f_n).exp();
                new[i + (k >> 1)] = old[i + k] + omega * old[i + k + stride];
                new[i + (k >> 1) + h] = old[i + k] - omega * old[i + k + stride];
            }
        }
        (old, new) = (new, old);
        swapped = !swapped;
        sublen <<= 1;
    }

    if swapped {
        cloned_v.copy_from_slice(&zero_v)
    }

    cloned_v
}

// Ergün, Funda. (1995, June). Testing multivariate linear functions: Overcoming the generator bottleneck.
// In Proc. Twenty-Seventh Ann. ACM Symp. Theory of Computing. (p. 407–416).
#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use crate::fft;
    use approx::assert_relative_eq;
    const ALPHA: f32 = 0.5;
    const BETA: f32 = 0.75;
    const N: usize = 32;

    #[test]
    fn milion_test() {
        let v: [f32; 65_536] = generate(|idx| f32::sin(idx as f32));
        let fft: [f32; 65_536] = fft::fft(&v); //Frequency size is 1Hz per bin
        assert_eq!(1.0f32, fft[3]);
    }

    #[test]
    fn linearity_holds() {
        let v: [f32; N] = generate(|idx| idx as f32);
        let e: [f32; N] = generate(|idx| (idx + 1usize) as f32);

        // FFT of sum
        let mut a_v = v.map(|v| v * ALPHA);
        let b_e = e.map(|v| v * BETA);
        sum_v(&mut a_v, &b_e);
        let sum = a_v;
        let fft_sum: [f32; N] = fft::fft(&sum);

        //Sum of FFT
        let mut a_fft_v: [f32; N] = fft::fft(&v).map(|v| ALPHA * v);
        let b_fft_e: [f32; N] = fft::fft(&e).map(|v| BETA * v);
        sum_v(&mut a_fft_v, &b_fft_e);
        let sum_fft = a_fft_v;

        array_assert_eq(fft_sum, sum_fft);
    }

    #[test]
    fn unit_impulse_holds() {
        let impulse = generate_impulse::<N, 0>();
        let fft_impulse: [f32; N] = fft::fft(&impulse);
        array_assert_eq(generate::<N>(|_| 1.0f32), fft_impulse);
    }

    #[test]
    fn time_shift_holds() {
        let a = generate(|idx| f32::cos((idx as f32) / 10.0));
        let b = generate(|idx| f32::cos(((idx + 1) as f32) / 10.0));
        let fft_a: [f32; N] = fft::fft(&a);
        let fft_b: [f32; N] = fft::fft(&b);
        assert_eq!(fft_a, fft_b);
        array_assert_eq(fft_a, fft_b);
    }

    fn sum_v(a: &mut [f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for idx in 0..a.len() {
            a[idx] += b[idx];
        }
    }

    fn generate<const S: usize>(generator: fn(usize) -> f32) -> [f32; S] {
        let mut v = [0.0f32; S];
        for (idx, val) in v.iter_mut().enumerate() {
            *val = generator(idx);
        }
        v
    }

    fn generate_impulse<const S: usize, const INSTANT: usize>() -> [f32; S] {
        generate(|idx| if idx == INSTANT { 1.0f32 } else { 0.0f32 })
    }

    fn array_assert_eq<A: approx::RelativeEq<A, Epsilon = f32> + Debug, const N: usize>(
        a: [A; N],
        b: [A; N],
    ) {
        (0..N).for_each(|idx| assert_relative_eq!(a[idx], b[idx], epsilon = 0.1f32))
    }
}
