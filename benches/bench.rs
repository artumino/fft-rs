use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
    sync::Arc,
};

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use fft::{
    allocators::{array::ArrayAllocator, boxed::BoxedAllocator},
    implementations::{cooley_tukey::OmegaCalculator, naive::ImgUnit, CooleyTukey, Naive},
    windows::{hanning::Hanning, Rect},
    Allocator, Implementation, WindowFunction,
};
use num_complex::{Complex32, ComplexFloat};

use random::Source;

pub fn criterion_benchmark(c: &mut Criterion) {
    run_cooley_bench_group::<Complex32, Rect>(c, "FFT<Complex32>");
    run_cooley_bench_group::<f32, Rect>(c, "FFT<f32>");
    run_cooley_bench_group::<Complex32, Hanning>(c, "FFT<Complex32> + Hanning");
    run_cooley_bench_group::<f32, Hanning>(c, "FFT<f32> + Hanning");
    run_naive_bench_group::<Complex32, Rect>(c, "Naive<Complex32>");
    run_naive_bench_group::<Complex32, Hanning>(c, "Naive<Complex32> + Hanning");
}

fn run_cooley_bench_group<T, W>(c: &mut Criterion, name: &'static str)
where
    W: WindowFunction<T>,
    T: Copy
        + Debug
        + Default
        + Add<Output = T>
        + OmegaCalculator<T>
        + Mul<<T as OmegaCalculator<T>>::TMul, Output = T>
        + Sub<Output = T>,
    [T]: Randomizable<T>,
{
    let mut group = c.benchmark_group(name);
    run_bench::<T, 1_048_576, CooleyTukey, W, BoxedAllocator>(&mut group);
    run_bench::<T, 65_536, CooleyTukey, W, BoxedAllocator>(&mut group);
    run_bench::<T, 1_024, CooleyTukey, W, BoxedAllocator>(&mut group);
    run_bench::<T, 1_024, CooleyTukey, W, ArrayAllocator>(&mut group);
    run_bench::<T, 512, CooleyTukey, W, BoxedAllocator>(&mut group);
    run_bench::<T, 512, CooleyTukey, W, ArrayAllocator>(&mut group);
    run_bench::<T, 32, CooleyTukey, W, BoxedAllocator>(&mut group);
    run_bench::<T, 32, CooleyTukey, W, ArrayAllocator>(&mut group);
    group.finish();
}

fn run_naive_bench_group<T, W>(c: &mut Criterion, name: &'static str)
where
    W: WindowFunction<T>,
    T: Copy
        + Debug
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<f32, Output = T>
        + ComplexFloat
        + ImgUnit,
    [T]: Randomizable<T>,
{
    let mut group = c.benchmark_group(name);
    run_bench::<T, 512, Naive, W, BoxedAllocator>(&mut group);
    run_bench::<T, 512, Naive, W, ArrayAllocator>(&mut group);
    run_bench::<T, 32, Naive, W, BoxedAllocator>(&mut group);
    run_bench::<T, 32, Naive, W, ArrayAllocator>(&mut group);
    group.finish();
}

fn run_bench<T, const N: usize, I, W, A>(c: &mut BenchmarkGroup<'_, WallTime>)
where
    I: Implementation<T, N, A>,
    A: Allocator<T, N>,
    W: WindowFunction<T>,
    T: Copy + Debug + Default,
    [T]: Randomizable<T>,
{
    let allocator_name = std::any::type_name::<A>().split("::").last().unwrap();
    let strategy_name = std::any::type_name::<I>().split("::").last().unwrap();
    let vec = generate::<T, N>();
    let mut out_spec = A::allocate();
    let boxed_engine = fft::Engine::<T, N, I, W, A>::new();
    c.bench_with_input(
        BenchmarkId::new(format!("{strategy_name}_{allocator_name}").as_str(), N),
        &N,
        |b, _| {
            b.iter(|| {
                let vec = vec.clone();
                boxed_engine.fft(black_box(vec.as_slice()), &mut out_spec)
            })
        },
    );
}

pub trait Randomizable<T> {
    fn randomize(&mut self, seed: u64);
}

fn generate<T: Default + Copy + Debug, const S: usize>() -> Arc<[T; S]>
where
    [T]: Randomizable<T>,
{
    let mut result: Box<[T; S]> = vec![Default::default(); S]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    result.randomize(12304);
    result.into()
}

impl Randomizable<f32> for [f32] {
    fn randomize(&mut self, seed: u64) {
        let mut rnd = random::default(seed);
        for val in self.iter_mut() {
            *val = rnd.read_f64() as f32;
        }
    }
}

impl Randomizable<Complex32> for [Complex32] {
    fn randomize(&mut self, seed: u64) {
        let mut rnd = random::default(seed);
        for val in self.iter_mut() {
            *val = Complex32::new(rnd.read_f64() as f32, rnd.read_f64() as f32);
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
