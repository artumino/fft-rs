# Rust FFT Library

A simple rust library that provides a no_std compatible FFT implementation.
It also allows to compose windowing functions with the FFT and select allocation methods.
Disable 'alloc' feature for environments without allocator (ArrayAllocator will become the only available allocator).
The 'alloc' feature is enabled by default and enables caching in some implementations.

## Example

### 1024 point FFT with Hanning window (on the stack)

```rust
let engine = Engine::<Complex32, 1024, CooleyTukey, Hanning, ArrayAllocator>::new();
let input = [Complex32::new(0.0, 0.0); 1024];
let mut spectrum = [Complex32::new(0.0, 0.0); 1024];
engine.fft(&input, &mut spectrum);
```

### 1024 point FFT without windowing function (on the heap)

```rust
let engine = Engine::<Complex32, 1024, CooleyTukey, Rect, BoxedAllocator>::new();
let input = vec![Complex32::new(0.0, 0.0); 1024];
let mut spectrum = vec![Complex32::new(0.0, 0.0); 1024].into_boxed_slice();
engine.fft(&input, &mut spectrum);
```
