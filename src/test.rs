// Ergün, Funda. (1995, June). Testing multivariate linear functions: Overcoming the generator bottleneck.
// In Proc. Twenty-Seventh Ann. ACM Symp. Theory of Computing. (p. 407–416).

use core::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Sub},
};

use approx::{assert_relative_eq, AbsDiffEq, RelativeEq};
use num_complex::Complex32;
use std::sync::Arc;

use crate::{
    allocators::boxed::BoxedAllocator,
    implementations::{cooley_tukey::OmegaCalculator, CooleyTukey},
    windows::Rect,
    Allocator, Engine, Implementation, WindowFunction,
};
const ALPHA: f32 = 0.5;
const BETA: f32 = 0.75;
const N: usize = 32;

pub trait EngineTest<T: Copy, const N: usize, A, W, I>
where
    A: Allocator<T, N>,
    I: Implementation<T, N, A>,
    W: WindowFunction<T>,
    T: Copy,
{
    fn engine() -> Engine<T, N, I, W, A>;
    fn allocate() -> A::Element;
}

pub struct TestFixture<T: Copy, const N: usize, A: Allocator<T, N>> {
    element_marker: PhantomData<T>,
    allocator_marker: PhantomData<A>,
}

impl<T, const N: usize, A: Allocator<T, N>> EngineTest<T, N, A, Rect, CooleyTukey>
    for TestFixture<T, N, A>
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + OmegaCalculator<T, TMul = f32>
        + Mul<f32, Output = T>,
{
    fn engine() -> Engine<T, N, CooleyTukey, Rect, A> {
        Engine::new()
    }

    fn allocate() -> A::Element {
        A::allocate()
    }
}

type ComplexTestFixture = TestFixture<Complex32, 32, BoxedAllocator>;
#[test]
fn linearity_holds() {
    let engine = ComplexTestFixture::engine();
    let v = generate::<N, Complex32>(|idx| Complex32::new(idx as f32, 0.0f32));
    let e = generate::<N, Complex32>(|idx| Complex32::new((idx + 1usize) as f32, 0.0f32));

    // FFT of sum
    let mut a_v = v.map(|v| v * ALPHA);
    let b_e = e.map(|v| v * BETA);
    sum_v(&mut a_v, b_e.into_iter());
    let sum = a_v;
    let mut fft_sum = ComplexTestFixture::allocate();
    engine.fft(&sum, &mut fft_sum);

    //Sum of FFT
    let mut a_fft_v = ComplexTestFixture::allocate();
    engine.fft(&v, &mut a_fft_v);
    mul_v(&mut a_fft_v, ALPHA);
    let mut e_fft_v = ComplexTestFixture::allocate();
    engine.fft(&e, &mut e_fft_v);
    sum_v(&mut a_fft_v, e_fft_v.iter_mut().map(|x| *x * BETA));
    let sum_fft = a_fft_v;

    array_assert_eq(fft_sum.as_ref(), sum_fft.as_ref(), 1e-1);
}

#[test]
fn unit_impulse_holds() {
    let engine = ComplexTestFixture::engine();
    let impulse = generate_impulse::<N, 0, Complex32>();
    let mut fft_impulse = ComplexTestFixture::allocate();
    engine.fft(&impulse, &mut fft_impulse);
    array_assert_eq(
        generate::<N, Complex32>(|_| Complex32::new(1.0f32, 0.0f32)).as_slice(),
        fft_impulse.as_ref(),
        1e-1,
    );
}

#[test]
fn time_shift_holds() {
    let engine = ComplexTestFixture::engine();
    let a = generate::<N, Complex32>(|idx| Complex32::new(f32::sin((idx as f32) / 10.0), 0.0f32));
    let b =
        generate::<N, Complex32>(|idx| Complex32::new(f32::sin(((idx + 1) as f32) / 10.0), 0.0f32));
    let mut fft_a = ComplexTestFixture::allocate();
    engine.fft(a.as_ref(), &mut fft_a);
    let mut fft_b = ComplexTestFixture::allocate();
    engine.fft(b.as_ref(), &mut fft_b);
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
where
    T: RelativeEq + Debug,
    <T as AbsDiffEq>::Epsilon: Copy,
{
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
