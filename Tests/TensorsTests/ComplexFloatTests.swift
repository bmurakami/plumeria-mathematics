import Testing
@testable import Tensors

@Test func ComplexFloat_plumeriaScalarSupport() {
    let value = ComplexFloat(1.2345678, -2.3456789)

    #expect(value.round() == ComplexFloat(1.234568, -2.345679))
    #expect(value.round(precision: 2) == ComplexFloat(1.23, -2.35))
    #expect(ComplexFloat.i == ComplexFloat(0.0, 1.0))
    #expect(ComplexFloat(1.0, 2.0).star == ComplexFloat(1.0, -2.0))
    #expect(ComplexFloat(3.0, 4.0).mod.isApproximatelyEqual(to: 5.0, relativeTolerance: 1e-6))
    #expect((Float(2.0) + ComplexFloat(1.0, -3.0)) == ComplexFloat(3.0, -3.0))
}

@Test func ComplexFloat_vectorReferenceAndBLASOperations() {
    checkComplexFloatVector(VectorDenseReference<ComplexFloat>.self)
    checkComplexFloatVector(VectorDenseBLAS<ComplexFloat>.self)
}

@Test func ComplexFloat_matrixReferenceAndBLASOperations() {
    checkComplexFloatMatrix(MatrixDenseReference<ComplexFloat>.self)
    checkComplexFloatMatrix(MatrixDenseBLAS<ComplexFloat>.self)
}

@Test func ComplexFloat_tensorReferenceAndBLASOperations() {
    checkComplexFloatTensor(TensorDenseReference<ComplexFloat>.self)
    checkComplexFloatTensor(TensorDenseBLAS<ComplexFloat>.self)
}

@Test func ComplexFloat_mixedTensorScalarArithmetic() {
    checkComplexFloatMixedReferenceTensorScalarArithmetic()
    checkComplexFloatMixedBLASTensorScalarArithmetic()
}

private func checkComplexFloatVector<V: PluVector>(_ type: V.Type) where V.S == ComplexFloat {
    let u = V([ComplexFloat(3.0, 4.0), ComplexFloat(0.0, 12.0)])
    let v = V([ComplexFloat(1.0, -1.0), ComplexFloat(2.0, 0.0)])

    #expect(u.magnitude().isApproximatelyEqual(to: 13.0, relativeTolerance: 1e-6))
    #expect((u + v).toArray() == [ComplexFloat(4.0, 3.0), ComplexFloat(2.0, 12.0)])
    #expect((u * ComplexFloat(0.0, 1.0)).toArray() == [ComplexFloat(-4.0, 3.0), ComplexFloat(-12.0, 0.0)])
    #expect((u / ComplexFloat(2.0, 0.0)).toArray() == [ComplexFloat(1.5, 2.0), ComplexFloat(0.0, 6.0)])
}

private func checkComplexFloatMatrix<M: PluMatrix>(_ type: M.Type) where M.S == ComplexFloat {
    let left = M([[ComplexFloat(1.0, 1.0), ComplexFloat(2.0, -1.0)],
                  [ComplexFloat(0.0, 3.0), ComplexFloat(-1.0, 0.0)]])
    let right = M([[ComplexFloat(1.0, 0.0), ComplexFloat(0.0, -1.0)],
                   [ComplexFloat(2.0, 0.0), ComplexFloat(1.0, 1.0)]])
    let vector = VectorDenseReference<ComplexFloat>([ComplexFloat(1.0, 0.0), ComplexFloat(2.0, 1.0)])

    #expect((left * vector).toArray() == [ComplexFloat(6.0, 1.0), ComplexFloat(-2.0, 2.0)])
    #expect((left * right).toArray() == [[ComplexFloat(5.0, -1.0), ComplexFloat(4.0, 0.0)],
                                        [ComplexFloat(-2.0, 3.0), ComplexFloat(2.0, -1.0)]])
}

private func checkComplexFloatTensor<T: TensorMultiplication & TensorArithmetic>(_ type: T.Type) where T.S == ComplexFloat {
    let leftValues = nestedComplexFloatArray([[ComplexFloat(1.0, 1.0), ComplexFloat(2.0, 0.0)],
                                              [ComplexFloat(0.0, 0.0), ComplexFloat(-1.0, 1.0)]])
    let rightValues = nestedComplexFloatArray([[ComplexFloat(1.0, 0.0), ComplexFloat(0.0, 1.0)],
                                               [ComplexFloat(2.0, -1.0), ComplexFloat(-1.0, 0.0)]])
    let left = T(leftValues)
    let right = T(rightValues)
    let product = multiply(left, [TensorIndex("i"), TensorIndex("j")], right, [TensorIndex("j"), TensorIndex("k")])

    #expect((left + left)[[0, 0]] == ComplexFloat(2.0, 2.0))
    #expect((left * ComplexFloat(0.0, 1.0))[[1, 1]] == ComplexFloat(-1.0, -1.0))
    #expect(product.shape == [2, 2])
    #expect(product[[0, 0]] == ComplexFloat(5.0, -1.0))
    #expect(product[[1, 0]] == ComplexFloat(-1.0, 3.0))
    #expect(product[[0, 1]] == ComplexFloat(-3.0, 1.0))
    #expect(product[[1, 1]] == ComplexFloat(1.0, -1.0))
}

private func nestedComplexFloatArray(_ values: [[ComplexFloat]]) -> TensorNestedArray<ComplexFloat> {
    .array(values.map { .array($0.map { .scalar($0) }) })
}

private func checkComplexFloatMixedReferenceTensorScalarArithmetic() {
    let real = TensorDenseReference<Float>(shape: [2], elements: [2.0, -1.0])
    let complex = TensorDenseReference<ComplexFloat>(shape: [2], elements: [ComplexFloat(2.0, -2.0),
                                                                             ComplexFloat(0.0, 4.0)])

    #expect((real * ComplexFloat(1.0, -1.0)).elements == [ComplexFloat(2.0, -2.0), ComplexFloat(-1.0, 1.0)])
    #expect((ComplexFloat(1.0, -1.0) * real).elements == (real * ComplexFloat(1.0, -1.0)).elements)
    #expect((real / ComplexFloat(1.0, -1.0)).elements == [ComplexFloat(1.0, 1.0), ComplexFloat(-0.5, -0.5)])
    #expect((complex * Float(2.0)).elements == [ComplexFloat(4.0, -4.0), ComplexFloat(0.0, 8.0)])
    #expect((Float(2.0) * complex).elements == (complex * Float(2.0)).elements)
    #expect((complex / Float(2.0)).elements == [ComplexFloat(1.0, -1.0), ComplexFloat(0.0, 2.0)])
}

private func checkComplexFloatMixedBLASTensorScalarArithmetic() {
    let real = TensorDenseBLAS<Float>(shape: [2], elements: [2.0, -1.0])
    let complex = TensorDenseBLAS<ComplexFloat>(shape: [2], elements: [ComplexFloat(2.0, -2.0),
                                                                       ComplexFloat(0.0, 4.0)])

    #expect((real * ComplexFloat(1.0, -1.0)).elements == [ComplexFloat(2.0, -2.0), ComplexFloat(-1.0, 1.0)])
    #expect((ComplexFloat(1.0, -1.0) * real).elements == (real * ComplexFloat(1.0, -1.0)).elements)
    #expect((real / ComplexFloat(1.0, -1.0)).elements == [ComplexFloat(1.0, 1.0), ComplexFloat(-0.5, -0.5)])
    #expect((complex * Float(2.0)).elements == [ComplexFloat(4.0, -4.0), ComplexFloat(0.0, 8.0)])
    #expect((Float(2.0) * complex).elements == (complex * Float(2.0)).elements)
    #expect((complex / Float(2.0)).elements == [ComplexFloat(1.0, -1.0), ComplexFloat(0.0, 2.0)])
}
