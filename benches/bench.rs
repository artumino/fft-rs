use std::sync::Arc;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use fft::fft::{
    allocators::{array::ArrayAllocator, boxed::BoxedAllocator},
    implementations::CooleyTukey,
    Allocator, Implementation,
};
use random::Source;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT<f32>");
    run_bench::<1_048_576, CooleyTukey, BoxedAllocator>(&mut group);
    run_bench::<65_536, CooleyTukey, BoxedAllocator>(&mut group);
    run_bench::<1_024, CooleyTukey, BoxedAllocator>(&mut group);
    run_bench::<1_024, CooleyTukey, ArrayAllocator>(&mut group);
    run_bench::<512, CooleyTukey, BoxedAllocator>(&mut group);
    run_bench::<512, CooleyTukey, ArrayAllocator>(&mut group);
    run_bench::<32, CooleyTukey, BoxedAllocator>(&mut group);
    run_bench::<32, CooleyTukey, ArrayAllocator>(&mut group);
    group.finish();
}

fn run_bench<const N: usize, I, A>(c: &mut BenchmarkGroup<'_, WallTime>)
where
    I: Implementation<f32, N, A>,
    A: Allocator<f32, N>,
{
    let allocator_name = std::any::type_name::<A>().split("::").last().unwrap();
    let strategy_name = std::any::type_name::<I>().split("::").last().unwrap();
    let vec = generate::<N>();
    let boxed_engine = fft::fft::Engine::<f32, N, I, A>::new();
    c.bench_with_input(
        BenchmarkId::new(format!("{strategy_name}_{allocator_name}").as_str(), N),
        &N,
        |b, _| b.iter(|| boxed_engine.fft(black_box(&vec))),
    );
}

fn generate<const S: usize>() -> Arc<[f32; S]> {
    let mut result: Box<[f32; S]> = vec![0.0f32; S].into_boxed_slice().try_into().unwrap();
    let mut rnd = random::default(12304);
    for val in result.iter_mut() {
        *val = rnd.read_f64() as f32;
    }
    result.into()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
