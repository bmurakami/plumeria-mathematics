#if canImport(Accelerate)
import AccelerateWrapper
#endif
import OpenBLASWrapper
import Numerics
import Tensors

public typealias DenseLinearSolver<M: PluMatrix, V: PluVector> = (M, V) -> V where M.S == V.S

public func solveLinearDense<M: PluMatrix, V: PluVector>(_ A: M, _ b: V, blasImplementation: BLAS = .default) -> V {
    solveLinearDenseBLAS(A, b, blasImplementation: blasImplementation)
}

public func solveLinearDense(
    _ A: MatrixDenseBLAS<Double>, _ b: VectorDenseBLAS<Double>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<Double> {
    solveLinearDenseBLAS(A, b, blasImplementation: blasImplementation)
}

public func solveLinearDense(
    _ A: MatrixDenseBLAS<Float>, _ b: VectorDenseBLAS<Float>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<Float> {
    solveLinearDenseBLAS(A, b, blasImplementation: blasImplementation)
}

public func solveLinearDense(
    _ A: MatrixDenseBLAS<ComplexDouble>, _ b: VectorDenseBLAS<ComplexDouble>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<ComplexDouble> {
    solveLinearDenseBLAS(A, b, blasImplementation: blasImplementation)
}

public func solveLinearDense(
    _ A: MatrixDenseBLAS<ComplexFloat>, _ b: VectorDenseBLAS<ComplexFloat>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<ComplexFloat> {
    solveLinearDenseBLAS(A, b, blasImplementation: blasImplementation)
}

public func solveLinearDenseReference<M: PluMatrix, V: PluVector>(_ A: M, _ b: V) -> V where M.S == V.S {
    precondition(A.rows == A.columns, "A must be square")
    precondition(A.columns == b.size, "Number of columns in A must equal size of b")
    let n = b.size
    precondition(n > 0, "Linear systems must be non-empty")
    var matrix = A.toArray()
    var rhs = b.toArray()
    for pivot in 0..<n {
        let pivotRow = rowWithLargestPivot(matrix, column: pivot)
        precondition(matrix[pivotRow][pivot].magnitude > .zero, "A must be non-singular")
        if pivotRow != pivot {
            matrix.swapAt(pivot, pivotRow)
            rhs.swapAt(pivot, pivotRow)
        }
        for row in (pivot + 1)..<n {
            let factor = matrix[row][pivot] / matrix[pivot][pivot]
            matrix[row][pivot] = .zero
            for column in (pivot + 1)..<n {
                matrix[row][column] -= factor * matrix[pivot][column]
            }
            rhs[row] -= factor * rhs[pivot]
        }
    }
    var x = Array(repeating: V.S.zero, count: n)
    for row in stride(from: n - 1, through: 0, by: -1) {
        var value = rhs[row]
        for column in (row + 1)..<n {
            value -= matrix[row][column] * x[column]
        }
        x[row] = value / matrix[row][row]
    }
    return V(x)
}

public func solveLinearDenseBLAS<M: PluMatrix, V: PluVector>(_ A: M, _ b: V, blasImplementation: BLAS = .default) -> V {
    precondition(A.rows == A.columns, "A must be square")
    precondition(A.columns == b.size, "Number of columns in A must equal size of b")
    let n = b.size
    var x: [V.S]

    switch V.S.self {
    case is Float.Type:
        var AArray = A.flatten() as! [Float]
        var bArray = b.toArray() as! [Float]

        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate:
            let _ = AccelerateOperations.sgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        #endif
        case .openBLAS:
            let _ = OpenBLASOperations.sgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        }
    case is Double.Type:
        var AArray = A.flatten() as! [Double]
        var bArray = b.toArray() as! [Double]

        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate:
            let _ = AccelerateOperations.dgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        #endif
        case .openBLAS:
            let _ = OpenBLASOperations.dgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        }
    case is ComplexDouble.Type:
        var AArray = BLASComplexStorage.interleaved(A.flatten() as! [ComplexDouble])
        var bArray = BLASComplexStorage.interleaved(b.toArray() as! [ComplexDouble])

        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate:
            let _ = AccelerateOperations.zgesv(Int32(n), &AArray, &bArray)
            x = BLASComplexStorage.complexValues(bArray) as! [V.S]
        #endif
        case .openBLAS:
            let _ = OpenBLASOperations.zgesv(Int32(n), &AArray, &bArray)
            x = BLASComplexStorage.complexValues(bArray) as! [V.S]
        }

    case is ComplexFloat.Type:
        var AArray = BLASComplexStorage.interleaved(A.flatten() as! [ComplexFloat])
        var bArray = BLASComplexStorage.interleaved(b.toArray() as! [ComplexFloat])

        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate:
            let _ = AccelerateOperations.cgesv(Int32(n), &AArray, &bArray)
            x = BLASComplexStorage.complexValues(bArray) as! [V.S]
        #endif
        case .openBLAS:
            let _ = OpenBLASOperations.cgesv(Int32(n), &AArray, &bArray)
            x = BLASComplexStorage.complexValues(bArray) as! [V.S]
        }

    default:
        fatalError("Unsupported scalar type")
    }

    return V(x)
}

public func solveLinearDenseBLAS(
    _ A: MatrixDenseBLAS<Double>, _ b: VectorDenseBLAS<Double>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<Double> {
    precondition(A.rows == A.columns, "A must be square")
    precondition(A.columns == b.size, "Number of columns in A must equal size of b")
    var matrix = A.flatten()
    var rhs = b.elements
    let info = solveDoubleLinearSystem(Int32(b.size), &matrix, &rhs, blasImplementation: blasImplementation)
    precondition(info == 0, "LAPACK linear solve failed with info \(info)")
    return VectorDenseBLAS(rhs)
}

public func solveLinearDenseBLAS(
    _ A: MatrixDenseBLAS<Float>, _ b: VectorDenseBLAS<Float>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<Float> {
    precondition(A.rows == A.columns, "A must be square")
    precondition(A.columns == b.size, "Number of columns in A must equal size of b")
    var matrix = A.flatten()
    var rhs = b.elements
    let info = solveFloatLinearSystem(Int32(b.size), &matrix, &rhs, blasImplementation: blasImplementation)
    precondition(info == 0, "LAPACK linear solve failed with info \(info)")
    return VectorDenseBLAS(rhs)
}

public func solveLinearDenseBLAS(
    _ A: MatrixDenseBLAS<ComplexDouble>, _ b: VectorDenseBLAS<ComplexDouble>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<ComplexDouble> {
    precondition(A.rows == A.columns, "A must be square")
    precondition(A.columns == b.size, "Number of columns in A must equal size of b")
    var matrix = BLASComplexStorage.interleaved(A.flatten())
    var rhs = BLASComplexStorage.interleaved(b.elements)
    let info = solveComplexDoubleLinearSystem(Int32(b.size), &matrix, &rhs, blasImplementation: blasImplementation)
    precondition(info == 0, "LAPACK linear solve failed with info \(info)")
    return VectorDenseBLAS(BLASComplexStorage.complexValues(rhs))
}

public func solveLinearDenseBLAS(
    _ A: MatrixDenseBLAS<ComplexFloat>, _ b: VectorDenseBLAS<ComplexFloat>, blasImplementation: BLAS = .default
) -> VectorDenseBLAS<ComplexFloat> {
    precondition(A.rows == A.columns, "A must be square")
    precondition(A.columns == b.size, "Number of columns in A must equal size of b")
    var matrix = BLASComplexStorage.interleaved(A.flatten())
    var rhs = BLASComplexStorage.interleaved(b.elements)
    let info = solveComplexFloatLinearSystem(Int32(b.size), &matrix, &rhs, blasImplementation: blasImplementation)
    precondition(info == 0, "LAPACK linear solve failed with info \(info)")
    return VectorDenseBLAS(BLASComplexStorage.complexValues(rhs))
}

private func solveFloatLinearSystem(
    _ n: Int32, _ matrix: inout [Float], _ rhs: inout [Float], blasImplementation: BLAS
) -> Int32 {
    switch blasImplementation {
    #if canImport(Accelerate)
    case .accelerate: AccelerateOperations.sgesv(n, &matrix, &rhs)
    #endif
    case .openBLAS: OpenBLASOperations.sgesv(n, &matrix, &rhs)
    }
}

private func solveDoubleLinearSystem(
    _ n: Int32, _ matrix: inout [Double], _ rhs: inout [Double], blasImplementation: BLAS
) -> Int32 {
    switch blasImplementation {
    #if canImport(Accelerate)
    case .accelerate: AccelerateOperations.dgesv(n, &matrix, &rhs)
    #endif
    case .openBLAS: OpenBLASOperations.dgesv(n, &matrix, &rhs)
    }
}

private func solveComplexFloatLinearSystem(
    _ n: Int32, _ matrix: inout [Float], _ rhs: inout [Float], blasImplementation: BLAS
) -> Int32 {
    switch blasImplementation {
    #if canImport(Accelerate)
    case .accelerate: AccelerateOperations.cgesv(n, &matrix, &rhs)
    #endif
    case .openBLAS: OpenBLASOperations.cgesv(n, &matrix, &rhs)
    }
}

private func solveComplexDoubleLinearSystem(
    _ n: Int32, _ matrix: inout [Double], _ rhs: inout [Double], blasImplementation: BLAS
) -> Int32 {
    switch blasImplementation {
    #if canImport(Accelerate)
    case .accelerate: AccelerateOperations.zgesv(n, &matrix, &rhs)
    #endif
    case .openBLAS: OpenBLASOperations.zgesv(n, &matrix, &rhs)
    }
}

private func rowWithLargestPivot<S: PluScalar>(_ matrix: [[S]], column: Int) -> Int {
    var row = column
    var pivot = matrix[column][column].magnitude
    for candidate in (column + 1)..<matrix.count {
        let magnitude = matrix[candidate][column].magnitude
        if magnitude > pivot {
            row = candidate
            pivot = magnitude
        }
    }
    return row
}
