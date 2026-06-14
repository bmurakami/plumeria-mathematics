# Benchmarks

All times are medians in milliseconds from release-mode runs on one Apple Silicon machine.

## Commands

```sh
swift run -c release PlumeriaBenchmarks
uv run --with numpy python Benchmarks/PythonNumPy/numpy_benchmarks.py
```

## Environment

- Swift: Apple Swift 6.3.2
- Python: 3.12.11
- NumPy: 2.4.6
- Platform: M1 iMac, 8 cores, macOS 26.5

## PluMath Reference vs. BLAS

| Operation | Reference | BLAS/LAPACK | Reference/BLAS |
| --- | ---: | ---: | ---: |
| vector add 10,000 | 1.5214 | **0.0026** | 578.2098 |
| vector scale 100,000 | 14.2040 | **0.0194** | 733.7443 |
| vector magnitude 100,000 | 13.5820 | **0.0164** | 826.7003 |
| matrix add 256x256 | 20.0685 | **0.0116** | 1730.0438 |
| matrix add 1,024x1,024 | 327.1950 | **0.1868** | 1751.6634 |
| matrix-vector 384x384 | 22.1599 | **0.0189** | 1174.0331 |
| matrix-vector 1,536x1,536 | 360.2930 | **0.5257** | 685.4015 |
| matrix-matrix 128x128 | 147.4689 | **0.0169** | 8717.7155 |
| determinant 96x96 | 44.2640 | **0.0279** | 1587.9447 |
| inverse 96x96 | 250.2594 | **0.0652** | 3837.8637 |
| linear solve 16x16 | 0.1113 | **0.0030** | 36.6107 |
| linear solve 64x64 | 6.0189 | **0.0205** | 293.0176 |
| linear solve 256x256 | 370.4053 | **0.2665** | 1390.1078 |
| tensor add 40x40x10 | 26.2750 | **0.0034** | 7658.8653 |
| tensor contraction 16x24x16,16x16 | 21.5922 | **0.0071** | 3048.0250 |
| tensor contraction 32x32x32,32x32 | 152.7592 | **0.0305** | 5015.4051 |

## PluMath BLAS vs. NumPy

| Operation | PluMath | NumPy |
| --- | ---: | ---: |
| vector add 10,000 | 0.0026 | **0.0019** |
| vector scale 100,000 | 0.0194 | **0.0178** |
| vector magnitude 100,000 | **0.0164** | 0.0169 |
| matrix add 256x256 | **0.0116** | 0.0122 |
| matrix add 1,024x1,024 | **0.1868** | 0.2036 |
| matrix-vector 384x384 | 0.0189 | **0.0111** |
| matrix-vector 1,536x1,536 | 0.5257 | **0.5244** |
| matrix-matrix 128x128 | **0.0169** | 0.0170 |
| determinant 96x96 | **0.0279** | 0.0315 |
| inverse 96x96 | 0.0652 | **0.0613** |
| linear solve 16x16 | **0.0030** | 0.0059 |
| linear solve 64x64 | **0.0205** | 0.0226 |
| linear solve 256x256 | 0.2665 | **0.2144** |
| tensor add 40x40x10 | **0.0034** | 0.0035 |
| tensor contraction 16x24x16,16x16 | **0.0071** | 0.0366 |
| tensor contraction 32x32x32,32x32 | **0.0305** | 0.2932 |

## Float and ComplexFloat

| Operation | Double | Float |
| --- | ---: | ---: |
| vector add 100,000 | 0.0181 | **0.0101** |
| magnitude 100,000 | 0.0161 | **0.0033** |
| matrix-vector 384x384 | 0.0190 | **0.0046** |
| matrix-matrix 128x128 | 0.0169 | **0.0058** |
| tensor contraction 16x24x16,16x16 | 0.0065 | **0.0052** |

| Operation | ComplexDouble | ComplexFloat |
| --- | ---: | ---: |
| vector add 100,000 | 0.0352 | **0.0182** |
| magnitude 100,000 | 0.0322 | **0.0065** |
| matrix-vector 384x384 | 0.1108 | **0.0167** |
| matrix-matrix 128x128 | 0.0750 | **0.0232** |
| tensor contraction 16x24x16,16x16 | 0.0128 | **0.0079** |

## NumPy Lazy Expression Check

NumPy does not appear to algebraically simplify repeated array terms in expressions such as
`D[1:-1, 1:-1] = A[1:-1, 1:-1] + B[1:-1, 1:-1] - B[1:-1, 1:-1]`. The expression costs more than
`A + B` and much more than copying `A` into the slice.

| Slice size | Copy `A` | `A + B` | `A + B - B` | `A + B - B` / Copy |
| --- | ---: | ---: | ---: | ---: |
| 254x254 | 0.0196 | 0.0642 | 0.0905 | 4.62 |
| 510x510 | 0.0773 | 0.2335 | 0.3456 | 4.47 |
| 1022x1022 | 0.3157 | 1.0649 | 1.4958 | 4.74 |
| 2046x2046 | 2.2758 | 4.8255 | 7.2002 | 3.16 |

## Notes

- Benchmarks are not correctness tests and do not run with `swift test`.
- Results depend on hardware, Swift version, NumPy version, BLAS implementation, and thermal state.
- Reference implementations are intentionally naive and math-like.
- BLAS/LAPACK implementations are meant to be fast and to support implementation comparisons.
- NumPy uses Accelerate on this macOS machine. For 256x256 linear solves, NumPy improves from about
  0.218 ms with C-order input to about 0.175 ms with Fortran-order input. PluMath is about
  0.26-0.28 ms after concrete dense solve overloads. A forced matrix/RHS copy costs only about
  0.035 ms, and direct standalone Swift calls to `dgesv_` were slower than PluMath, so the remaining
  gap appears to be in the exact LAPACK interface path rather than generic dispatch.
