use criterion::{black_box, criterion_group, criterion_main, Criterion};
use random::Source;

pub fn criterion_benchmark(c: &mut Criterion) {
    let vec_65k: [f32; 65_536] = generate();
    c.bench_function("fft array 65536", |b| {
        b.iter(|| fft::fft::fft(black_box(&vec_65k)))
    });
}

fn generate<const S: usize>() -> [f32; S] {
    let mut rnd = random::default(12304);
    [0.0f32; S].map(|_| rnd.read_f64() as f32)
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
