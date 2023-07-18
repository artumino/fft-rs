# Rust FFT Library

A simple rust library that provides a no_std compatible FFT implementation.
It also allows to compose windowing functions with the FFT and select allocation methods.
Disable 'alloc' feature for environments without allocator (ArrayAllocator will become the only available allocator).

## Example

### 1024 point FFT with Hanning window

```rust
let engine = Engine<Complex<f32>, 1024, CooleyTukey, Hanning, BoxedAllocator>::new();
let input = [Complex::new(0.0, 0.0); 1024];
let mut spectrum = [Complex::new(0.0, 0.0); 1024];
engine.fft(&mut input, &mut spectrum);
```
