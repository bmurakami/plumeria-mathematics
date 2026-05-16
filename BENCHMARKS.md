# Benchmarks

Plumeria benchmarks compare reference and BLAS-backed implementations for representative vector,
matrix, and tensor operations.

Run benchmarks explicitly in release mode:

```bash
swift run -c release PlumeriaBenchmarks
```

Benchmarks are not correctness tests and do not run with `swift test`. Timings depend on hardware,
operating system, Swift version, BLAS backend, and thermal state.

Current benchmark groups:

- Vector addition, scalar multiplication, and magnitude.
- Matrix addition, matrix-vector multiplication, and matrix-matrix multiplication.
- Tensor elementwise addition and tensor contraction.
