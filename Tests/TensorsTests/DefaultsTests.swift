import Testing
@testable import Tensors

@Test func DefaultMatrix_aliasUsesRecommendedImplementation() {
    let A = Matrix<Double>([[1.0, 2.0], [3.0, 4.0]])
    let v = Vector<Double>([2.0, 3.0])

    #expect(A.toArray() == [[1.0, 2.0], [3.0, 4.0]])
    #expect((A * v).toArray() == [8.0, 18.0])
}

@Test func DenseMatrixD_matchesDefaultMatrixSpelling() {
    let A = Matrix<Double>([[1.0, 2.0], [3.0, 4.0]])
    let B = DenseMatrixDouble([[1.0, 2.0], [3.0, 4.0]])

    #expect(A.toArray() == B.toArray())
}

@Test func ReferenceMatrixD_usesReferenceImplementationSpelling() {
    let A = ReferenceMatrixDouble([[1.0, 2.0], [3.0, 4.0]])
    let v = ReferenceVectorDouble([2.0, 3.0])

    #expect(A.toArray() == [[1.0, 2.0], [3.0, 4.0]])
    #expect((A * v).toArray() == [8.0, 18.0])
}
