import Testing
@testable import Tensors

@Test func Matrix_physicsProperties() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0],
                                          [3.0, 4.0]])

    #expect(matrix.t.toArray() == [[1.0, 3.0], [2.0, 4.0]])
    #expect(matrix.tr == 5.0)
    #expect(matrix.det == -2.0)
}

@Test func Matrix_identityAndInverse() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0],
                                          [3.0, 4.0]])
    let inverse = matrix.inverse()
    let identity = MatrixDenseBLAS<Double>.identity(size: 2)

    #expect(identity.toArray() == [[1.0, 0.0], [0.0, 1.0]])
    expectClose(inverse, [[-2.0, 1.0], [1.5, -0.5]], tolerance: 1e-12)
    expectCloseToIdentity(matrix * inverse, tolerance: 1e-12)
}

@Test func Matrix_BLAS_lapackDeterminantAndInverse() {
    let matrix = MatrixDenseBLAS<Double>([[4.0, 7.0, 2.0],
                                          [3.0, 6.0, 1.0],
                                          [2.0, 5.0, 1.0]])

    #expect(matrix.det.isClose(to: 3.0, relativeTolerance: 1e-12))
    expectCloseToIdentity(matrix * matrix.inverse(), tolerance: 1e-12)
}

@Test func Matrix_BLAS_lapackSingularDeterminant() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0],
                                          [2.0, 4.0]])

    #expect(matrix.det == 0.0)
}

@Test func Matrix_BLAS_lapackFloatDeterminantAndInverse() {
    let matrix = MatrixDenseBLAS<Float>([[4.0, 7.0, 2.0],
                                         [3.0, 6.0, 1.0],
                                         [2.0, 5.0, 1.0]])

    #expect(matrix.det.isClose(to: 3.0, relativeTolerance: 1e-5))
    expectCloseToIdentity(matrix * matrix.inverse(), tolerance: 1e-5)
}

@Test func Matrix_BLAS_lapackComplexDeterminantAndInverse() {
    let matrix = MatrixDenseBLAS<ComplexDouble>([[ComplexDouble(1.0, 1.0), ComplexDouble(2.0, 0.0)],
                                                 [ComplexDouble(0.0, 0.0), ComplexDouble(3.0, -1.0)]])

    #expect((matrix.det - ComplexDouble(4.0, 2.0)).mod.isClose(to: 0.0, relativeTolerance: 1e-12))
    expectCloseToIdentity(matrix * matrix.inverse(), tolerance: 1e-12)
}

@Test func Matrix_BLAS_lapackComplexFloatDeterminantAndInverse() {
    let matrix = MatrixDenseBLAS<ComplexFloat>([[ComplexFloat(1.0, 1.0), ComplexFloat(2.0, 0.0)],
                                                [ComplexFloat(0.0, 0.0), ComplexFloat(3.0, -1.0)]])

    #expect((matrix.det - ComplexFloat(4.0, 2.0)).mod.isClose(to: 0.0, relativeTolerance: 1e-5))
    expectCloseToIdentity(matrix * matrix.inverse(), tolerance: 1e-5)
}

#if canImport(Accelerate)
@Test func Matrix_BLAS_lapackImplementationSelection() {
    let values = [4.0, 3.0, 2.0, 7.0, 6.0, 5.0, 2.0, 1.0, 1.0]
    let accelerate = MatrixDenseBLAS(rows: 3, columns: 3, values: values, blasImplementation: .accelerate)
    let openBLAS = MatrixDenseBLAS(rows: 3, columns: 3, values: values, blasImplementation: .openBLAS)

    #expect(accelerate.det.isClose(to: openBLAS.det, relativeTolerance: 1e-12))
    #expect(accelerate.inverse().isClose(to: openBLAS.inverse(), relativeTolerance: 1e-12))
}
#endif

extension MatrixImplementation {
    func checkEigenRealEigenvalues() {
        switch self {
        case .reference: verifyEigenRealEigenvalues(MatrixDenseReference<Double>.self)
        case .blas: verifyEigenRealEigenvalues(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkEigenComplexEigenvalues() {
        switch self {
        case .reference: verifyEigenComplexEigenvalues(MatrixDenseReference<Double>.self)
        case .blas: verifyEigenComplexEigenvalues(MatrixDenseBLAS<Double>.self)
        }
    }
}

@Test(arguments: MatrixImplementation.allCases)
func Matrix_eigenRealEigenvalues(implementation: MatrixImplementation) {
    implementation.checkEigenRealEigenvalues()
}

@Test(arguments: MatrixImplementation.allCases)
func Matrix_eigenComplexEigenvalues(implementation: MatrixImplementation) {
    implementation.checkEigenComplexEigenvalues()
}

private func verifyEigenRealEigenvalues<M: MatrixEigen>(_ type: M.Type) where M.Eigenvalue == ComplexDouble {
    let matrix = M([[2.0, 0.0],
                    [0.0, 3.0]])
    let eigen = matrix.eigen()

    expectEigenvalues(eigen.values, [ComplexDouble(2.0, 0.0), ComplexDouble(3.0, 0.0)])
    expectEigenpair(matrix, eigen, j: 0)
    expectEigenpair(matrix, eigen, j: 1)
}

private func verifyEigenComplexEigenvalues<M: MatrixEigen>(_ type: M.Type) where M.Eigenvalue == ComplexDouble {
    let matrix = M([[0.0, -1.0],
                    [1.0, 0.0]])
    let eigen = matrix.eigen()

    expectEigenvalues(eigen.values, [ComplexDouble(0.0, 1.0), ComplexDouble(0.0, -1.0)])
    expectEigenpair(matrix, eigen, j: 0)
    expectEigenpair(matrix, eigen, j: 1)
}

private func expectEigenvalues(_ values: [ComplexDouble], _ expected: [ComplexDouble]) {
    #expect(values.count == expected.count)
    for value in expected {
        #expect(values.contains { $0.isClose(to: value, relativeTolerance: 1e-12) })
    }
}

private func expectEigenpair<M: MatrixEigen>(
    _ matrix: M, _ eigen: Eigen<M.Eigenvalue, M.Eigenvectors>, j: Int
) where M.Eigenvalue == ComplexDouble {
    let vector = VectorDenseReference<ComplexDouble>((0..<matrix.rows).map { eigen.vectors[$0, j] })
    let left = complexMatrix(matrix) * vector
    let right = VectorDenseReference<ComplexDouble>((0..<vector.size).map { eigen.values[j] * vector[$0] })

    #expect(left.isClose(to: right, relativeTolerance: 1e-12))
}

private func complexMatrix<M: MatrixEigen>(_ matrix: M) -> MatrixDenseBLAS<ComplexDouble> {
    MatrixDenseBLAS<ComplexDouble>((0..<matrix.rows).map { i in
        (0..<matrix.columns).map { j in ComplexDouble(matrix[i, j], 0.0) }
    })
}

private func expectClose(_ matrix: MatrixDenseBLAS<Double>, _ expected: [[Double]], tolerance: Double) {
    for i in 0..<matrix.rows {
        for j in 0..<matrix.columns {
            #expect(abs(matrix[i, j] - expected[i][j]) <= tolerance)
        }
    }
}

private func expectCloseToIdentity(_ matrix: MatrixDenseBLAS<Double>, tolerance: Double) {
    for i in 0..<matrix.rows {
        for j in 0..<matrix.columns {
            let expected = i == j ? 1.0 : 0.0
            #expect(abs(matrix[i, j] - expected) <= tolerance)
        }
    }
}

private func expectCloseToIdentity(_ matrix: MatrixDenseBLAS<Float>, tolerance: Float) {
    for i in 0..<matrix.rows {
        for j in 0..<matrix.columns {
            let expected: Float = i == j ? 1.0 : 0.0
            #expect(abs(matrix[i, j] - expected) <= tolerance)
        }
    }
}

private func expectCloseToIdentity(_ matrix: MatrixDenseBLAS<ComplexDouble>, tolerance: Double) {
    for i in 0..<matrix.rows {
        for j in 0..<matrix.columns {
            let expected = i == j ? ComplexDouble(1.0, 0.0) : .zero
            #expect((matrix[i, j] - expected).mod <= tolerance)
        }
    }
}

private func expectCloseToIdentity(_ matrix: MatrixDenseBLAS<ComplexFloat>, tolerance: Float) {
    for i in 0..<matrix.rows {
        for j in 0..<matrix.columns {
            let expected = i == j ? ComplexFloat(1.0, 0.0) : .zero
            #expect((matrix[i, j] - expected).mod <= tolerance)
        }
    }
}
