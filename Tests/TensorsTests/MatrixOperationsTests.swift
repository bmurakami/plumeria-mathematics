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

@Test func Matrix_eigenRealEigenvalues() {
    let matrix = MatrixDenseBLAS<Double>([[2.0, 0.0],
                                          [0.0, 3.0]])
    let eigen = matrix.eigen()

    #expect(eigen.values == [Complex(2.0, 0.0), Complex(3.0, 0.0)])
    expectEigenpair(matrix, eigen, column: 0)
    expectEigenpair(matrix, eigen, column: 1)
}

@Test func Matrix_eigenComplexEigenvalues() {
    let matrix = MatrixDenseBLAS<Double>([[0.0, -1.0],
                                          [1.0, 0.0]])
    let eigen = matrix.eigen()

    #expect(eigen.values[0].isApproximatelyEqual(to: Complex(0.0, 1.0)))
    #expect(eigen.values[1].isApproximatelyEqual(to: Complex(0.0, -1.0)))
    expectEigenpair(matrix, eigen, column: 0)
    expectEigenpair(matrix, eigen, column: 1)
}

private func expectEigenpair(_ matrix: MatrixDenseBLAS<Double>, _ eigen: Eigen, column: Int) {
    let vector = VectorDenseReference<Complex>((0..<matrix.rows).map { eigen.vectors[$0, column] })
    let left = complexMatrix(matrix) * vector
    let right = VectorDenseReference<Complex>((0..<vector.size).map { eigen.values[column] * vector[$0] })

    #expect(left.isApproximatelyEqual(to: right, relativeTolerance: 1e-12))
}

private func complexMatrix(_ matrix: MatrixDenseBLAS<Double>) -> MatrixDenseBLAS<Complex> {
    MatrixDenseBLAS<Complex>((0..<matrix.rows).map { row in
        (0..<matrix.columns).map { column in Complex(matrix[row, column], 0.0) }
    })
}
