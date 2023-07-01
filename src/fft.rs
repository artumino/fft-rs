use core::marker::PhantomData;

use self::implementations::cooley::Cooley;

pub mod allocators;
pub mod implementations;

pub trait Allocator<T, const N: usize> {
    type Element: AsMut<[T]> + AsRef<[T]> + Sized;
    fn allocate() -> Self::Element;
}

pub trait Implementation<T, const N: usize, A>
where
    A: Allocator<T, N>,
{
    fn fft(v: &[T; N]) -> A::Element;
}

pub struct Engine<T, const N: usize, I, A>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    T: Copy,
{
    impl_marker: PhantomData<I>,
    allocator_marker: PhantomData<A>,
    element_marker: PhantomData<T>,
}

type DefaultImpl = Cooley;

#[cfg(feature = "alloc")]
type DefaultAllocator = allocators::boxed::BoxedAllocator;

#[cfg(not(feature = "alloc"))]
type DefaultAllocator = allocators::array::ArrayAllocator;

impl<const N: usize> Default for Engine<f32, N, DefaultImpl, DefaultAllocator> {
    fn default() -> Engine<f32, N, DefaultImpl, DefaultAllocator> {
        Engine {
            impl_marker: PhantomData,
            allocator_marker: PhantomData,
            element_marker: PhantomData,
        }
    }
}

impl<T, const N: usize, I, A> Engine<T, N, I, A>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    T: Copy,
{
    pub fn new() -> Engine<T, N, I, A> {
        Engine {
            impl_marker: PhantomData,
            allocator_marker: PhantomData,
            element_marker: PhantomData,
        }
    }

    pub fn fft(&self, v: &[T; N]) -> <A as Allocator<T, N>>::Element {
        <I as Implementation<T, N, A>>::fft(v)
    }
}

// Ergün, Funda. (1995, June). Testing multivariate linear functions: Overcoming the generator bottleneck.
// In Proc. Twenty-Seventh Ann. ACM Symp. Theory of Computing. (p. 407–416).
#[cfg(test)]
mod test {
    use std::fmt::Debug;

    use std::sync::Arc;
    use approx::assert_relative_eq;
    const ALPHA: f32 = 0.5;
    const BETA: f32 = 0.75;
    const N: usize = 32;

    #[test]
    fn linearity_holds() {
        let engine = crate::fft::Engine::default();
        let v = generate::<N>(|idx| idx as f32);
        let e = generate::<N>(|idx| (idx + 1usize) as f32);

        // FFT of sum
        let mut a_v = v.map(|v| v * ALPHA);
        let b_e = e.map(|v| v * BETA);
        sum_v(&mut a_v, b_e.into_iter());
        let sum = a_v;
        let fft_sum = engine.fft(&sum);

        //Sum of FFT
        let mut a_fft_v = engine.fft(&v);
        mul_v(&mut a_fft_v, ALPHA);
        sum_v(&mut a_fft_v, engine.fft(&e).iter_mut().map(|x| *x * BETA));
        let sum_fft = a_fft_v;

        array_assert_eq(fft_sum.as_ref(), sum_fft.as_ref());
    }

    #[test]
    fn unit_impulse_holds() {
        let engine = crate::fft::Engine::default();
        let impulse = generate_impulse::<N, 0>();
        let fft_impulse = engine.fft(&impulse);
        array_assert_eq(generate::<N>(|_| 1.0f32).as_slice(), fft_impulse.as_ref());
    }

    #[test]
    fn time_shift_holds() {
        let engine = crate::fft::Engine::default();
        let a = generate::<N>(|idx| f32::cos((idx as f32) / 10.0));
        let b = generate::<N>(|idx| f32::cos(((idx + 1) as f32) / 10.0));
        let fft_a = engine.fft(a.as_ref());
        let fft_b = engine.fft(b.as_ref());
        assert_eq!(fft_a, fft_b);
        array_assert_eq(fft_a.as_ref(), fft_b.as_ref());
    }

    fn sum_v(a: &mut [f32], b: impl Iterator<Item = f32>) {
        for (idx, val) in b.enumerate() {
            a[idx] += val;
        }
    }

    fn mul_v(a: &mut [f32], alpha: f32) {
        for val in a {
            *val *= alpha;
        }
    }

    fn generate<const S: usize>(generator: fn(usize) -> f32) -> Arc<[f32; S]> {
        let mut v: Box<[f32; S]> = vec![0.0f32; S].into_boxed_slice().try_into().unwrap();
        for (idx, val) in v.iter_mut().enumerate() {
            *val = generator(idx);
        }
        v.into()
    }

    fn generate_impulse<const S: usize, const INSTANT: usize>() -> Arc<[f32; S]> {
        generate(|idx| if idx == INSTANT { 1.0f32 } else { 0.0f32 })
    }

    fn array_assert_eq<A: approx::RelativeEq<A, Epsilon = f32> + Debug>(a: &[A], b: &[A]) {
        assert_eq!(a.len(), b.len());
        (0..a.len()).for_each(|idx| assert_relative_eq!(a[idx], b[idx], epsilon = 0.1f32))
    }
}
