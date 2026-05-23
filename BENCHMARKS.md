# Benchmarks

## Commands
```
swift run -c release PlumeriaBenchmarks
uv run --with numpy python Benchmarks/PythonNumPy/numpy_benchmarks.py
```

## Environment

- Swift: Apple Swift 6.3.2
- Python: 3.12.11
- NumPy: 2.4.6
- Platform: arm64 macOS

## Findings

- NumPy is extremely fast for dense vectorized operations. On this machine, NumPy matrix-vector and matrix-matrix
  multiplication are faster than the current PluMath BLAS wrappers at the benchmarked sizes.
- PluMath reference implementations are intentionally naive and are much slower than PluMath BLAS implementations.
- PluMath BLAS wins strongly over PluMath reference for matrix multiplication, determinants, inverses, vector magnitude,
  and tensor contraction.
- Float and ComplexFloat show only modest gains in PluMath, while NumPy shows larger Float32 and Complex64 gains for
  some operations.
- Accelerate and OpenBLAS are close on this machine. Neither dominates across all measured operations.
- The current PluMath matrix-add benchmark is not comparable to NumPy matrix addition after lazy matrix expressions:
  it reads two result cells, so it measures expression construction and scalar lookup rather than full materialization.
- Fused slice arithmetic is a separate case. For the wave-style stencil update, PluMath fuses the expression into one
  assignment loop and beat NumPy on this machine.

## Selected Results

| Operation | PluMath BLAS median | NumPy median | Note |
| --- | ---: | ---: | --- |
| vector magnitude 100,000 | 1.1955 ms | 0.0176 ms | NumPy much faster |
| matrix-vector 384x384 | 1.8077 ms | 0.0199 ms | NumPy much faster |
| matrix-matrix 128x128 | 0.5990 ms | 0.0350 ms | NumPy much faster |
| determinant 96x96 | 0.1356 ms | 0.0330 ms | NumPy faster |
| inverse 96x96 | 0.1762 ms | 0.1187 ms | NumPy faster |
| tensor contraction 16x24x16,16x16 | 2.5366 ms | 0.0747 ms | NumPy much faster |
| tensor contraction 32x32x32,32x32 | 13.6268 ms | 0.5538 ms | NumPy much faster |
| fused slice update 300x300 | 0.1599 ms | 0.4312 ms | PluMath faster |

## Follow-Up

- Fix PluMath benchmarks that became lazy-expression benchmarks by forcing full materialization where needed.
- Investigate BLAS wrapper overhead and data-layout conversion costs for matrix-vector, matrix-matrix, and tensor
  contraction.
- Keep the wave-stencil benchmark separate from these general benchmarks because it measures fused slice arithmetic,
  not ordinary eager array operations.
