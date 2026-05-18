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
    #expect(inverse.toArray() == [[-2.0, 1.0], [1.5, -0.5]])
    #expect((matrix * inverse).isApproximatelyEqual(to: identity, relativeTolerance: 1e-12))
}

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

private func verifyEigenRealEigenvalues<M: MatrixEigen>(_ type: M.Type) {
    let matrix = M([[2.0, 0.0],
                    [0.0, 3.0]])
    let eigen = matrix.eigen()

    expectEigenvalues(eigen.values, [Complex(2.0, 0.0), Complex(3.0, 0.0)])
    expectEigenpair(matrix, eigen, column: 0)
    expectEigenpair(matrix, eigen, column: 1)
}

private func verifyEigenComplexEigenvalues<M: MatrixEigen>(_ type: M.Type) {
    let matrix = M([[0.0, -1.0],
                    [1.0, 0.0]])
    let eigen = matrix.eigen()

    expectEigenvalues(eigen.values, [Complex(0.0, 1.0), Complex(0.0, -1.0)])
    expectEigenpair(matrix, eigen, column: 0)
    expectEigenpair(matrix, eigen, column: 1)
}

private func expectEigenvalues(_ values: [Complex], _ expected: [Complex]) {
    #expect(values.count == expected.count)
    for value in expected {
        #expect(values.contains { $0.isApproximatelyEqual(to: value, relativeTolerance: 1e-12) })
    }
}

private func expectEigenpair<M: MatrixEigen>(_ matrix: M, _ eigen: Eigen<M.Eigenvectors>, column: Int) {
    let vector = VectorDenseReference<Complex>((0..<matrix.rows).map { eigen.vectors[$0, column] })
    let left = complexMatrix(matrix) * vector
    let right = VectorDenseReference<Complex>((0..<vector.size).map { eigen.values[column] * vector[$0] })

    #expect(left.isApproximatelyEqual(to: right, relativeTolerance: 1e-12))
}

private func complexMatrix<M: MatrixEigen>(_ matrix: M) -> MatrixDenseBLAS<Complex> {
    MatrixDenseBLAS<Complex>((0..<matrix.rows).map { row in
        (0..<matrix.columns).map { column in Complex(matrix[row, column], 0.0) }
    })
}
