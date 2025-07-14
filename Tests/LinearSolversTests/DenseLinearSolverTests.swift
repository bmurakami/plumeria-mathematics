import Foundation
import Testing
import Tensors
@testable import LinearSolvers

@Test func solveLinearDense_exampleUsage() throws {
    // Solution: x=1, y=2, z=3
    //
    // 2x - y + z = 3
    // x + 2y - z = 2
    // x + y - 4z = -9
    
    let A = MatrixDenseBLAS([[2.0, -1, 1], [1, 2, -1], [1, 1, -4]])
    let b = VectorDenseReference([3.0, 2, -9])
    let v_actual = solveLinearDense(A, b)
    let v_expected = VectorDenseReference([1.0, 2, 3])
    
    #expect(v_actual.approximatelyEquals(v_expected))
}

func solveLinearDense_correctness_smallMatrices<MatrixType: PluMatrix>(matrixType: MatrixType.Type)
        where MatrixType.S == Double {
    let sizes = [2, 3, 10]
    for n in sizes {
        let A: MatrixType = makeMatrix(size: n)
        let b = makeVector(size: n)
        let v = solveLinearDense(A, b)
        
        #expect((A * v).approximatelyEquals(b, tolerance: 1e-13))
    }
}

@Test func solveLinearDense_correctness_smallMatrices_reference() {
    srand48(42)
    solveLinearDense_correctness_smallMatrices(matrixType: MatrixDenseReference<Double>.self)
}

@Test func solveLinearDense_correctness_smallMatrices_BLAS() {
    srand48(42)
    solveLinearDense_correctness_smallMatrices(matrixType: MatrixDenseBLAS<Double>.self)
}

func solveLinearDense_correctness_largeMatrices<MatrixType: PluMatrix>(matrixType: MatrixType.Type)
        where MatrixType.S == Double {
    let sizes = [100, 500]
    for n in sizes {
        let A: MatrixDenseBLAS<Double> = makeMatrix(size: n)
        let b = makeVector(size: n)
        let v = solveLinearDense(A, b)
        
        #expect((A * v).approximatelyEquals(b, tolerance: 1e-8))
    }
}

@Test func solveLinearDense_correctness_largeMatrices_reference() {
    srand48(42)
    solveLinearDense_correctness_largeMatrices(matrixType: MatrixDenseReference<Double>.self)
}

@Test func solveLinearDense_correctness_largeMatrices_BLAS() {
    srand48(42)
    solveLinearDense_correctness_largeMatrices(matrixType: MatrixDenseBLAS<Double>.self)
}

func randomNumber() -> Double {
    let sign = drand48() > 0.5 ? 1.0 : -1.0
    return sign * (drand48() * 99.0 + 1.0)
}

func makeVector(size: Int) -> VectorDenseReference<Double> {
    var v = [Double]()
    for _ in 0..<size {
        v.append(randomNumber())
    }
    return VectorDenseReference(v)
}

func makeMatrix<M: PluMatrix>(size: Int) -> M
    where M.S == Double {
    var matrix = [[Double]]()
    for _ in 0..<size {
        var row = [Double]()
        for _ in 0..<size {
            row.append(randomNumber())
        }
        matrix.append(row)
    }
    return M(matrix)
}
