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

    #expect(v_actual.isClose(to: v_expected))
}

@Test func solveLinearDenseReference_exampleUsage() throws {
    let A = MatrixDenseReference([[2.0, -1, 1], [1, 2, -1], [1, 1, -4]])
    let b = VectorDenseReference([3.0, 2, -9])
    let actual = solveLinearDenseReference(A, b)
    let expected = VectorDenseReference([1.0, 2, 3])

    #expect(actual.isClose(to: expected))
}

func solveLinearDense_correctness_smallMatrices<MatrixType: PluMatrix>(matrixType: MatrixType.Type) where MatrixType.S == Double {
    let sizes = [2, 3, 10]
    for n in sizes {
        let A: MatrixType = makeMatrix(size: n)
        let b = makeVector(size: n)
        let v = solveLinearDense(A, b)

        #expect((A * v).isClose(to: b, relativeTolerance: 1e-12))
    }
}

@Test func solveLinearDense_correctness_smallMatrices_reference() {
    let A2 = MatrixDenseReference([[2.0, 1.0], [1.0, -1.0]])
    let b2 = VectorDenseReference([4.0, -1.0])
    let x2 = VectorDenseReference([1.0, 2.0])
    let A3 = MatrixDenseReference([[2.0, -1, 1], [1, 2, -1], [1, 1, -4]])
    let b3 = VectorDenseReference([3.0, 2, -9])
    let x3 = VectorDenseReference([1.0, 2, 3])

    #expect(solveLinearDenseReference(A2, b2).isClose(to: x2))
    #expect(solveLinearDenseReference(A3, b3).isClose(to: x3))
}

@Test func solveLinearDenseReference_pivotsRows() {
    let A = MatrixDenseReference([[0.0, 2.0], [1.0, 1.0]])
    let b = VectorDenseReference([-2.0, 2.0])
    let expected = VectorDenseReference([3.0, -1.0])

    #expect(solveLinearDenseReference(A, b).isClose(to: expected))
}

@Test func solveLinearDense_correctness_smallMatrices_BLAS() {
    srand48(42)
    solveLinearDense_correctness_smallMatrices(matrixType: MatrixDenseBLAS<Double>.self)
}

@Test func solveLinearDense_floatBLAS() {
    let A = MatrixDenseBLAS<Float>([[2.0, -1.0, 1.0], [1.0, 2.0, -1.0], [1.0, 1.0, -4.0]])
    let b = VectorDenseBLAS<Float>([3.0, 2.0, -9.0])
    let expected = VectorDenseBLAS<Float>([1.0, 2.0, 3.0])
    let actual = solveLinearDense(A, b)

    #expect(actual.isClose(to: expected, relativeTolerance: 1e-5))

    #if canImport(Accelerate)
    let accelerate = solveLinearDense(A, b, blasImplementation: .accelerate)
    #expect(accelerate.isClose(to: expected, relativeTolerance: 1e-5))
    #endif
}

@Test func solveLinearDense_complexBLAS() {
    let A = MatrixDenseBLAS<ComplexDouble>([[ComplexDouble(1.0, 0.0), ComplexDouble(0.0, 1.0)],
                                      [ComplexDouble(2.0, -1.0), ComplexDouble(1.0, 1.0)]])
    let b = VectorDenseReference<ComplexDouble>([ComplexDouble(2.0, 3.0), ComplexDouble(6.0, 2.0)])
    let expected = VectorDenseReference<ComplexDouble>([ComplexDouble(1.0, 1.0), ComplexDouble(2.0, -1.0)])
    let actual = solveLinearDense(A, b)

    #expect(actual.isClose(to: expected, relativeTolerance: 1e-12))

    #if canImport(Accelerate)
    let accelerate = solveLinearDense(A, b, blasImplementation: .accelerate)
    #expect(accelerate.isClose(to: expected, relativeTolerance: 1e-12))
    #endif
}

@Test func solveLinearDense_complexFloatBLAS() {
    let A = MatrixDenseBLAS<ComplexFloat>([[ComplexFloat(1.0, 0.0), ComplexFloat(0.0, 1.0)],
                                           [ComplexFloat(2.0, -1.0), ComplexFloat(1.0, 1.0)]])
    let b = VectorDenseBLAS<ComplexFloat>([ComplexFloat(2.0, 3.0), ComplexFloat(6.0, 2.0)])
    let expected = VectorDenseBLAS<ComplexFloat>([ComplexFloat(1.0, 1.0), ComplexFloat(2.0, -1.0)])
    let actual = solveLinearDense(A, b)

    #expect(actual.isClose(to: expected, relativeTolerance: 1e-5))

    #if canImport(Accelerate)
    let accelerate = solveLinearDense(A, b, blasImplementation: .accelerate)
    #expect(accelerate.isClose(to: expected, relativeTolerance: 1e-5))
    #endif
}

@Test func solveLinearDenseReference_complex() {
    let A = MatrixDenseReference<ComplexDouble>([[ComplexDouble(1.0, 0.0), ComplexDouble(0.0, 1.0)],
                                                 [ComplexDouble(2.0, -1.0), ComplexDouble(1.0, 1.0)]])
    let b = VectorDenseReference<ComplexDouble>([ComplexDouble(2.0, 3.0), ComplexDouble(6.0, 2.0)])
    let expected = VectorDenseReference<ComplexDouble>([ComplexDouble(1.0, 1.0), ComplexDouble(2.0, -1.0)])
    let actual = solveLinearDenseReference(A, b)

    #expect(actual.isClose(to: expected, relativeTolerance: 1e-12))
}

func solveLinearDenseBLAS_correctness_largeMatrices() {
    let sizes = [100, 500]
    for n in sizes {
        let A: MatrixDenseBLAS<Double> = makeMatrix(size: n)
        let b = makeVector(size: n)
        let v = solveLinearDense(A, b)

        #expect((A * v).isClose(to: b, relativeTolerance: 1e-8))
    }
}

@Test func solveLinearDense_correctness_largeMatrices_BLAS() {
    srand48(42)
    solveLinearDenseBLAS_correctness_largeMatrices()
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
