use crate::fft::Allocator;

pub struct ArrayAllocator;
impl<const N: usize> Allocator<f32, N> for ArrayAllocator {
    type Element = [f32; N];

    fn allocate() -> Self::Element {
        [0.0f32; N]
    }
}

// Ergün, Funda. (1995, June). Testing multivariate linear functions: Overcoming the generator bottleneck.
// In Proc. Twenty-Seventh Ann. ACM Symp. Theory of Computing. (p. 407–416).
#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use approx::assert_relative_eq;

    use crate::fft::{allocators::array::ArrayAllocator, implementations::cooley::Cooley};
    const ALPHA: f32 = 0.5;
    const BETA: f32 = 0.75;
    const N: usize = 32;

    fn test_engine<const N: usize>() -> crate::fft::Engine<f32, N, Cooley, ArrayAllocator> {
        crate::fft::Engine::new()
    }

    #[test]
    fn milion_test() {
        let v: [f32; 65_536] = generate_impulse::<65_536, 0>();
        let engine = test_engine();
        let fft: [f32; 65_536] = engine.fft(&v); //Frequency size is 1Hz per bin
        assert_eq!(1.0f32, fft[3]);
    }

    #[test]
    fn linearity_holds() {
        let engine = test_engine();
        let v: [f32; N] = generate(|idx| idx as f32);
        let e: [f32; N] = generate(|idx| (idx + 1usize) as f32);

        // FFT of sum
        let mut a_v = v.map(|v| v * ALPHA);
        let b_e = e.map(|v| v * BETA);
        sum_v(&mut a_v, &b_e);
        let sum = a_v;
        let fft_sum: [f32; N] = engine.fft(&sum);

        //Sum of FFT
        let mut a_fft_v: [f32; N] = engine.fft(&v).map(|v| ALPHA * v);
        let b_fft_e: [f32; N] = engine.fft(&e).map(|v| BETA * v);
        sum_v(&mut a_fft_v, &b_fft_e);
        let sum_fft = a_fft_v;

        array_assert_eq(fft_sum, sum_fft);
    }

    #[test]
    fn unit_impulse_holds() {
        let engine = test_engine();
        let impulse = generate_impulse::<N, 0>();
        let fft_impulse: [f32; N] = engine.fft(&impulse);
        array_assert_eq(generate::<N>(|_| 1.0f32), fft_impulse);
    }

    #[test]
    fn time_shift_holds() {
        let engine = test_engine();
        let a = generate(|idx| f32::cos((idx as f32) / 10.0));
        let b = generate(|idx| f32::cos(((idx + 1) as f32) / 10.0));
        let fft_a: [f32; N] = engine.fft(&a);
        let fft_b: [f32; N] = engine.fft(&b);
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
