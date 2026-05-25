# Plumeria Mathematics

Plumeria Mathematics is a young Swift mathematics library for tensors, matrices, vectors, and linear algebra.

The goal is twofold:

- NumPy-like convenience and performance from Swift.
- Swappable implementations for academic comparison, such as reference arithmetic, Accelerate, and OpenBLAS.

This project is immature and still changing. It is useful for experiments, demos, and API shaping, but it is not yet a
stable scientific-computing foundation.

## Use

Add Plumeria Mathematics as a SwiftPM dependency:

```swift
.package(url: "https://github.com/bmurakami/plumeria-mathematics.git", branch: "main")
```

Then depend on the product:

```swift
.product(name: "PlumeriaMathematics", package: "plumeria-mathematics")
```

and import it:

```swift
import PlumeriaMathematics
```

## Examples

Dense vectors and matrices use the recommended BLAS/LAPACK-backed implementation by default:

```swift
let v = Vector<Double>([3, 4, 12])
let A = Matrix<Double>([
    [1, 2, 3],
    [4, 5, 6]
])

let length = v.magnitude()
let AT = A.t
```

Implementations can also be chosen explicitly:

```swift
let fast = MatrixDenseBLAS<Double>([
    [1, 2],
    [3, 4]
])

let reference = MatrixDenseReference<Double>([
    [1, 2],
    [3, 4]
])
```

Matrix operations use compact math names:

```swift
let I = Matrix<Double>.identity(size: 3)
let determinant = fast.det
let inverse = fast.inverse()
let trace = fast.tr
let eigen = fast.eigen()
```

Tensors support nested-array initialization and physics-style index notation:

```swift
let A = TensorDenseBLAS<Double>([
    [[1, 2], [3, 4], [5, 6]],
    [[0, -1], [2, 3], [-2, 1]]
])

let B = permute(A, "ijk -> jik")
let C = multiply(A, B, "ijk, jik")
```

Slices can be read, combined, and assigned:

```swift
let interior = range(1..<299)

u[0, interior, interior] =
    alpha[interior, interior] * (
        u[1, range(0..<298), interior] +
        u[1, range(2..<300), interior] +
        u[1, interior, range(0..<298)] +
        u[1, interior, range(2..<300)] -
        4 * u[1, interior, interior]
    )
```

Complex values use Swift Numerics under the hood:

```swift
let i = ComplexDouble.i
let z = 1.2 + 3.4 * i

z.star
z.mod
z.arg
```

## Benchmarks

Example release-mode median times on one Apple Silicon machine:

| Operation | PluMath reference | PluMath BLAS/LAPACK | NumPy |
| --- | ---: | ---: | ---: |
| vector magnitude, 100,000 | 13.5820 ms | **0.0164 ms** | 0.0169 ms |
| matrix-matrix multiply, 128x128 | 147.4689 ms | **0.0169 ms** | 0.0170 ms |
| determinant, 96x96 | 44.2640 ms | **0.0279 ms** | 0.0315 ms |
| tensor contraction, 32x32x32 by 32x32 | 152.7592 ms | **0.0305 ms** | 0.2932 ms |

Run local benchmarks with:

```sh
swift run -c release PlumeriaBenchmarks
uv run --with numpy python Benchmarks/PythonNumPy/numpy_benchmarks.py
```
