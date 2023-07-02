use core::marker::PhantomData;

use self::implementations::CooleyTukey;

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

type DefaultImpl = CooleyTukey;

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
    use core::{
        fmt::Debug,
        ops::{AddAssign, MulAssign},
    };

    use approx::{assert_relative_eq, RelativeEq, AbsDiffEq};
    use num_complex::Complex32;
    use std::sync::Arc;

    use super::{allocators::boxed::BoxedAllocator, implementations::CooleyTukey, Engine};
    const ALPHA: f32 = 0.5;
    const BETA: f32 = 0.75;
    const N: usize = 32;

    fn test_engine<const N: usize>() -> Engine<Complex32, N, CooleyTukey, BoxedAllocator> {
        Engine::new()
    }

    #[test]
    fn linearity_holds() {
        let engine = test_engine();
        let v = generate::<N, Complex32>(|idx| Complex32::new(idx as f32, 0.0f32));
        let e = generate::<N, Complex32>(|idx| Complex32::new((idx + 1usize) as f32, 0.0f32));

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

        array_assert_eq(fft_sum.as_ref(), sum_fft.as_ref(), 1e-1);
    }

    #[test]
    fn unit_impulse_holds() {
        let engine = test_engine();
        let impulse = generate_impulse::<N, 0, Complex32>();
        let fft_impulse = engine.fft(&impulse);
        array_assert_eq(
            generate::<N, Complex32>(|_| Complex32::new(1.0f32, 0.0f32)).as_slice(),
            fft_impulse.as_ref(),
            1e-1,
        );
    }

    #[test]
    fn time_shift_holds() {
        let engine = test_engine();
        let a =
            generate::<N, Complex32>(|idx| Complex32::new(f32::sin((idx as f32) / 10.0), 0.0f32));
        let b = generate::<N, Complex32>(|idx| {
            Complex32::new(f32::sin(((idx + 1) as f32) / 10.0), 0.0f32)
        });
        let fft_a = engine.fft(a.as_ref());
        let fft_b = engine.fft(b.as_ref());
        assert_eq!(fft_a, fft_b);
        array_assert_eq(fft_a.as_ref(), fft_b.as_ref(), 1e-1);
    }

    fn sum_v<T>(a: &mut [T], b: impl Iterator<Item = T>)
    where
        T: AddAssign<T> + Copy,
    {
        for (idx, val) in b.enumerate() {
            a[idx] += val;
        }
    }

    fn mul_v<T, TMul>(a: &mut [T], alpha: TMul)
    where
        T: MulAssign<TMul> + Copy,
        TMul: Copy,
    {
        for val in a {
            *val *= alpha;
        }
    }

    fn generate<const S: usize, T>(generator: fn(usize) -> T) -> Arc<[T; S]>
    where
        T: Default + Copy + Debug,
    {
        let mut v: Box<[T; S]> = vec![T::default(); S].into_boxed_slice().try_into().unwrap();
        for (idx, val) in v.iter_mut().enumerate() {
            *val = generator(idx);
        }
        v.into()
    }

    fn generate_impulse<const S: usize, const INSTANT: usize, T>() -> Arc<[T; S]>
    where
        T: Default + Copy + Debug + Impulse + Zeroed,
    {
        generate(|idx| {
            if idx == INSTANT {
                T::impulse()
            } else {
                T::zeroed()
            }
        })
    }

    fn array_assert_eq<T>(a: &[T], b: &[T], epsilon: <T as AbsDiffEq>::Epsilon)
        where T : RelativeEq + Debug,
              <T as AbsDiffEq>::Epsilon : Copy {
        assert_eq!(a.len(), b.len());
        (0..a.len()).for_each(|idx| assert_relative_eq!(a[idx], b[idx], epsilon = epsilon));
    }

    trait Impulse {
        fn impulse() -> Self;
    }

    trait Zeroed {
        fn zeroed() -> Self;
    }

    impl Impulse for Complex32 {
        #[inline]
        fn impulse() -> Self {
            Complex32::new(1.0f32, 0.0f32)
        }
    }

    impl Zeroed for Complex32 {
        #[inline]
        fn zeroed() -> Self {
            Complex32::new(0.0f32, 0.0f32)
        }
    }
}
