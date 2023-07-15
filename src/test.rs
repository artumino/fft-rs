// Linearity + Impulse inspired by:
// Ergün, Funda. (1995, June). Testing multivariate linear functions: Overcoming the generator bottleneck.
// In Proc. Twenty-Seventh Ann. ACM Symp. Theory of Computing. (p. 407–416).

use core::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Sub},
};

use approx::{assert_relative_eq, AbsDiffEq, RelativeEq};
use num_complex::{Complex32, ComplexFloat};
use num_traits::{One, Zero};

use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

use crate::{
    allocators::boxed::BoxedAllocator,
    implementations::{naive::ImgUnit, Naive},
    windows::Rect,
    Allocator, Engine, Implementation, Scalar, WindowFunction,
};
const ALPHA: Scalar = 0.5;
const BETA: Scalar = 0.75;
const N: usize = 32;

pub(crate) trait EngineTest<T, const N: usize, A, W, I>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    W: WindowFunction<T>,
    T: Copy + Mul<Scalar, Output = T> + ImgUnit + ComplexFloat,
{
    fn naive_engine() -> Engine<T, N, Naive, W, A>;
    fn test_engine() -> Engine<T, N, I, Rect, A>;
    fn allocate() -> A::Element;
}

pub(crate) struct TestFixture<
    T: Copy,
    const N: usize,
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
> {
    element_marker: PhantomData<T>,
    allocator_marker: PhantomData<A>,
    implementation_marker: PhantomData<I>,
}

impl<T, const N: usize, A: Allocator<T, N>, I: Implementation<T, N, A>> EngineTest<T, N, A, Rect, I>
    for TestFixture<T, N, A, I>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Scalar, Output = T> + ImgUnit + ComplexFloat,
{
    fn naive_engine() -> Engine<T, N, Naive, Rect, A> {
        Engine::new()
    }

    fn test_engine() -> Engine<T, N, I, Rect, A> {
        Engine::new()
    }

    fn allocate() -> A::Element {
        A::allocate()
    }
}

impl<T, const N: usize, A: Allocator<T, N>, I: Implementation<T, N, A>> TestFixture<T, N, A, I>
where
    T: Copy
        + Add<Output = T>
        + AddAssign<T>
        + Sub<Output = T>
        + MulAssign<T>
        + MulAssign<Scalar>
        + Mul<Scalar, Output = T>
        + ImgUnit
        + ComplexFloat
        + Default
        + Debug
        + One
        + Zero
        + RelativeEq,
    T::Epsilon: Copy + From<Scalar>,
    Standard: Distribution<T>,
{
    pub fn impulse_test() {
        let engine = Self::test_engine();
        let impulse = generate_impulse::<N, 0, T>();
        let mut fft_impulse = A::allocate();
        engine.fft(impulse.as_slice(), &mut fft_impulse);
        array_assert_eq(
            generate::<N, T>(|_| T::one()).as_slice(),
            fft_impulse.as_ref(),
            T::Epsilon::from(1e-1),
        );
    }

    pub fn linearity_test() {
        let engine = Self::test_engine();
        let v = generate::<N, T>(|idx| T::one() * idx as Scalar);
        let e = generate::<N, T>(|idx| T::one() * (idx + 1usize) as Scalar);

        // FFT of sum
        let mut a_v = v.map(|v| v * ALPHA);
        let b_e = e.map(|v| v * BETA);
        sum_v(&mut a_v, b_e.into_iter());
        let sum = a_v;
        let mut fft_sum = A::allocate();
        engine.fft(sum.as_slice(), &mut fft_sum);

        //Sum of FFT
        let mut a_fft_v = A::allocate();
        engine.fft(v.as_slice(), &mut a_fft_v);
        mul_v(a_fft_v.as_mut(), ALPHA);
        let mut e_fft_v = A::allocate();
        engine.fft(e.as_slice(), &mut e_fft_v);
        sum_v(
            a_fft_v.as_mut(),
            e_fft_v.as_mut().iter_mut().map(|x| *x * BETA),
        );
        let sum_fft = a_fft_v;

        array_assert_eq(fft_sum.as_ref(), sum_fft.as_ref(), T::Epsilon::from(1e-1));
    }

    const GROUND_TEST_SEEDS: [u64; 3] = [1234, 495611, 38596722];
    pub fn ground_truth_test() {
        let engine = Self::test_engine();
        let naive_engine = Self::naive_engine();

        for seed in Self::GROUND_TEST_SEEDS.iter() {
            let mut rng = StdRng::seed_from_u64(*seed);
            let v = (0..N).map(|_| rng.gen()).collect::<Vec<_>>();
            println!("v: {:?}", v);
            let mut fft_v = A::allocate();
            engine.fft(v.as_slice(), &mut fft_v);
            println!("fft: {:?}", fft_v.as_ref());
            let mut naive_fft_v = A::allocate();
            naive_engine.fft(v.as_slice(), &mut naive_fft_v);
            println!("naive_fft: {:?}", naive_fft_v.as_ref());
            array_assert_eq(naive_fft_v.as_ref(), fft_v.as_ref(), T::Epsilon::from(1e-1));
        }
    }
}

pub(crate) type ComplexTestFixture<I> = TestFixture<Complex32, N, BoxedAllocator, I>;

pub(crate) fn sum_v<T>(a: &mut [T], b: impl Iterator<Item = T>)
where
    T: AddAssign<T> + Copy,
{
    for (idx, val) in b.enumerate() {
        a[idx] += val;
    }
}

pub(crate) fn mul_v<T, TMul>(a: &mut [T], alpha: TMul)
where
    T: MulAssign<TMul> + Copy,
    TMul: Copy,
{
    for val in a {
        *val *= alpha;
    }
}

pub(crate) fn generate<const S: usize, T>(generator: fn(usize) -> T) -> Arc<[T; S]>
where
    T: Default + Copy + Debug,
{
    let mut v: Box<[T; S]> = vec![T::default(); S].into_boxed_slice().try_into().unwrap();
    for (idx, val) in v.iter_mut().enumerate() {
        *val = generator(idx);
    }
    v.into()
}

pub(crate) fn generate_impulse<const S: usize, const INSTANT: usize, T>() -> Arc<[T; S]>
where
    T: Default + Copy + Debug + One + Zero,
{
    generate(|idx| if idx == INSTANT { T::one() } else { T::zero() })
}

pub(crate) fn array_assert_eq<T>(a: &[T], b: &[T], epsilon: <T as AbsDiffEq>::Epsilon)
where
    T: RelativeEq + Debug,
    <T as AbsDiffEq>::Epsilon: Copy,
{
    assert_eq!(a.len(), b.len());
    (0..a.len()).for_each(|idx| {
        println!("Evaluating idx: {idx}");
        assert_relative_eq!(a[idx], b[idx], epsilon = epsilon)
    });
}
