# PlumeriaMathematics

PlumeriaMathematics is a Swift package for tensors, vectors, matrices, and dense linear algebra. The current focus is a small, explicit public SDK with swappable implementations underneath it.

The package exposes the umbrella module:

```swift
import PlumeriaMathematics
```

That re-exports the `Tensors` and `LinearSolvers` modules.

## Requirements

- Swift 6.0 or newer
- macOS 15 or newer
- OpenBLAS support through the bundled `COpenBLAS` system library target
- Accelerate support where available

## Installation

Add the package to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/bmurakami/plumeria-mathematics.git", branch: "main")
]
```

Then add the product to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "PlumeriaMathematics", package: "plumeria-mathematics")
    ]
)
```

## Quick Start

Use `Matrix` and `Vector` for the recommended default implementations:

```swift
import PlumeriaMathematics

let A = Matrix<Double>([
    [1.0, 2.0],
    [3.0, 4.0]
])

let x = Vector<Double>([2.0, 3.0])
let y = A * x

print(y.toArray()) // [8.0, 18.0]
```

Solve a dense linear system:

```swift
let A = Matrix<Double>([
    [2.0, -1.0, 1.0],
    [1.0, 2.0, -1.0],
    [1.0, 1.0, -4.0]
])

let b = Vector<Double>([3.0, 2.0, -9.0])
let solution = solveLinearDense(A, b)

print(solution.toArray()) // [1.0, 2.0, 3.0]
```

## Choosing Implementations

The default public types are aliases over implementation-backed wrappers:

```swift
public typealias Vector<S: PluScalar> = VectorBase<VectorDenseReference<S>>
public typealias Matrix<S: PluScalar> = MatrixBase<MatrixDenseBLAS<S>>
```

For everyday use:

```swift
let A = Matrix<Double>([[1.0, 2.0], [3.0, 4.0]])
let v = Vector<Double>([2.0, 3.0])
```

For explicit implementation choices:

```swift
let dense = DenseMatrixD([[1.0, 2.0], [3.0, 4.0]])
let reference = ReferenceMatrixD([[1.0, 2.0], [3.0, 4.0]])
```

Current aliases:

```swift
DenseVectorD
DenseMatrixD
ReferenceVectorD
ReferenceMatrixD
```

Mixed backends are intentionally not implicit. Convert explicitly when you want values to move between implementations.

## Indexing

Vectors and matrices keep the usual mathematical indexing:

```swift
let value = A[1, 0]
let entry = x[0]
```

`FlatTensor` also supports general tensor indexing:

```swift
tensor[indices: 1, 2]
tensor[[1, 2]]
```

The variadic tensor form is labeled as `indices:` so it does not conflict with matrix syntax like `matrix[1, 2]` on stricter Swift compilers.

## Storage Order

Flat tensor storage is column-major.

```swift
let stored = tensor.flatten(order: .columnMajor)
let exported = tensor.flatten(order: .rowMajor)
```

For matrices, `flatten()` and `flatten(columnMajorOrder: true)` return column-major storage. Row-major flattening is available for interop, display, and export.

## Architecture

The public protocols separate mathematical behavior from storage layout:

```text
PluTensor
├── PluVector
└── PluMatrix

FlatTensor
```

`PluVector` and `PluMatrix` describe vector and matrix behavior. `FlatTensor` describes flat column-major storage. A type may conform to both, but neither protocol implies the other.

Current concrete implementations:

```text
VectorDenseReference: PluVector, FlatTensor
MatrixDenseBLAS:      PluMatrix, FlatTensor
MatrixDenseReference: PluMatrix
```

Current public wrappers:

```text
VectorBase<Implementation: PluVector>: PluVector
MatrixBase<Implementation: PluMatrix>: PluMatrix
```

The wrapper types let the SDK expose convenient defaults while preserving the ability to evaluate or teach with alternate implementations.

## Current Scope

Implemented today:

- Dense vectors
- Dense matrices
- Flat column-major tensor storage protocol
- Matrix-vector multiplication
- Dense linear solves for supported scalar/backend combinations
- Approximate equality for vectors and matrices

Intentionally deferred:

- Slicing
- Sparse matrices
- A public general `Tensor` default type
- Rank 3 and higher tensor operation coverage
- Complex BLAS/LAPACK operation coverage
- Implicit mixed-backend operations

## Development

Run the test suite:

```sh
swift test
```

The tests cover the public defaults, reference implementations, BLAS-backed dense matrices, flat tensor indexing, storage-order conversion, and dense linear solvers.
