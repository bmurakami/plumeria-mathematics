import Testing
@testable import Tensors

@Test func Float_plumeriaScalarSupport() {
    let value: Float = 1.23456

    #expect(value.round(precision: 2) == 1.23)
    #expect((value / 2.0).isClose(to: 0.61728, relativeTolerance: 1e-6))
}

@Test func Float_vectorReferenceAndBLASOperations() {
    checkFloatVector(VectorDenseReference<Float>.self)
    checkFloatVector(VectorDenseBLAS<Float>.self)
}

@Test func Float_matrixReferenceAndBLASOperations() {
    checkFloatMatrix(MatrixDenseReference<Float>.self)
    checkFloatMatrix(MatrixDenseBLAS<Float>.self)
}

@Test func Float_tensorReferenceAndBLASOperations() {
    checkFloatTensor(TensorDenseReference<Float>.self)
    checkFloatTensor(TensorDenseBLAS<Float>.self)
}

private func checkFloatVector<V: PluVector>(_ type: V.Type) where V.S == Float {
    let u = V([3.0, 4.0, 12.0])
    let v = V([2.0, -1.0, 3.0])

    #expect(u.magnitude().isClose(to: 13.0, relativeTolerance: 1e-6))
    #expect(u.dot(v) == 38.0)
    #expect((u + v).toArray() == [5.0, 3.0, 15.0])
    #expect((u * 2.0).toArray() == [6.0, 8.0, 24.0])
    #expect((u / 2.0).toArray() == [1.5, 2.0, 6.0])
}

private func checkFloatMatrix<M: PluMatrix>(_ type: M.Type) where M.S == Float {
    let left = M([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let right = M([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    let vector = VectorDenseReference<Float>([2.0, 3.0, 4.0])

    #expect((left * vector).toArray() == [20.0, 47.0])
    #expect((left * right).toArray() == [[58.0, 64.0], [139.0, 154.0]])
    #expect((left + left).toArray() == [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
}

private func checkFloatTensor<T: TensorMultiplication & TensorArithmetic>(_ type: T.Type) where T.S == Float {
    let leftValues: TensorNestedArray<Float> = [[1.0, 2.0, 0.0], [-1.0, 3.0, 1.0]]
    let rightValues: TensorNestedArray<Float> = [[2.0, 1.0], [0.0, -1.0], [3.0, 2.0]]
    let left = T(leftValues)
    let right = T(rightValues)
    let product = multiply(left, [TensorIndex("i"), TensorIndex("j")], right, [TensorIndex("j"), TensorIndex("k")])

    #expect((left + left)[[0, 1]] == 4.0)
    #expect((left * 2.0)[[1, 0]] == -2.0)
    #expect(product.shape == [2, 2])
    #expect(product[[0, 0]] == 2.0)
    #expect(product[[1, 0]] == 1.0)
    #expect(product[[0, 1]] == -1.0)
    #expect(product[[1, 1]] == -2.0)
}
