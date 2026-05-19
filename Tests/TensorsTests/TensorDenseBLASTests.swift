import Testing
@testable import Tensors

private func tensor(_ values: [[[Double]]]) -> TensorDenseBLAS<Double> {
    let shape = [values.count, values[0].count, values[0][0].count]
    var tensor = TensorDenseBLAS<Double>(shape: shape, initialValue: 0.0)
    for i in 0..<shape[0] {
        for j in 0..<shape[1] {
            for k in 0..<shape[2] { tensor[i, j, k] = values[i][j][k] }
        }
    }
    return tensor
}

@Test func TensorDenseBLAS_storesColumnMajorElements() {
    var tensor = TensorDenseBLAS<Double>(shape: [2, 3], elements: [1.0, -1.0, 2.0, -2.0, 3.0, -3.0])

    #expect(tensor.shape == [2, 3])
    #expect(tensor.rank == 2)
    #expect(tensor.elements == [1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    #expect(tensor[[0, 0]] == 1.0)
    #expect(tensor[[1, 0]] == -1.0)
    #expect(tensor[0, 2] == 3.0)

    tensor[[1, 1]] = 0.0

    #expect(tensor.elements == [1.0, -1.0, 2.0, 0.0, 3.0, -3.0])
}

@Test func TensorDenseBLAS_mixedScalarArithmetic() {
    let realScalar = TensorDenseBLAS<Double>(shape: [], elements: [2.0])
    let realRank3 = TensorDenseBLAS<Double>(shape: [2, 2, 2], elements: [1.0, 0.0, -1.0, 2.0, 3.0, -2.0, 0.0, 1.0])
    let complexRank3 = TensorDenseBLAS<ComplexDouble>(shape: [2, 2, 2], elements: [
        ComplexDouble(1.0, -1.0), ComplexDouble(0.0, 2.0), ComplexDouble(-1.0, 0.0), ComplexDouble(2.0, 1.0),
        ComplexDouble(3.0, -2.0), ComplexDouble(-2.0, 1.0), ComplexDouble(0.0, -1.0), ComplexDouble(1.0, 0.0)
    ])
    #expect((realScalar * ComplexDouble(1.0, -1.0)).elements == [ComplexDouble(2.0, -2.0)])
    #expect((realRank3 * ComplexDouble(0.0, 2.0)).elements == [
        ComplexDouble(0.0, 2.0), ComplexDouble(0.0, 0.0), ComplexDouble(0.0, -2.0), ComplexDouble(0.0, 4.0),
        ComplexDouble(0.0, 6.0), ComplexDouble(0.0, -4.0), ComplexDouble(0.0, 0.0), ComplexDouble(0.0, 2.0)
    ])
    #expect((ComplexDouble(0.0, 2.0) * realRank3).elements == (realRank3 * ComplexDouble(0.0, 2.0)).elements)
    #expect((complexRank3 * 2.0).elements == [
        ComplexDouble(2.0, -2.0), ComplexDouble(0.0, 4.0), ComplexDouble(-2.0, 0.0), ComplexDouble(4.0, 2.0),
        ComplexDouble(6.0, -4.0), ComplexDouble(-4.0, 2.0), ComplexDouble(0.0, -2.0), ComplexDouble(2.0, 0.0)
    ])
    #expect((complexRank3 / 2.0).elements == [
        ComplexDouble(0.5, -0.5), ComplexDouble(0.0, 1.0), ComplexDouble(-0.5, 0.0), ComplexDouble(1.0, 0.5),
        ComplexDouble(1.5, -1.0), ComplexDouble(-1.0, 0.5), ComplexDouble(0.0, -0.5), ComplexDouble(0.5, 0.0)
    ])
}

@Test func TensorDenseBLAS_outerProduct() {
    let left = TensorDenseBLAS<Double>(shape: [2], elements: [2.0, -1.0])
    let right = TensorDenseBLAS<Double>(shape: [3], elements: [3.0, 0.0, -2.0])
    let product = left.times(right, contract: [])

    #expect(product.shape == [2, 3])
    #expect(product.elements == [6.0, -3.0, 0.0, 0.0, -4.0, 2.0])
}

@Test func TensorDenseBLAS_dotProductReturnsRankZeroTensor() {
    let left = TensorDenseBLAS<Double>(shape: [3], elements: [1.0, 2.0, -1.0])
    let right = TensorDenseBLAS<Double>(shape: [3], elements: [3.0, -1.0, 2.0])
    let product = left.times(right, contract: [(left: 0, right: 0)])

    #expect(product.shape == [])
    #expect(product.rank == 0)
    #expect(product.elements == [-1.0])
    #expect(product[[]] == -1.0)
}

@Test func TensorDenseBLAS_matrixVectorContraction() {
    let matrix = TensorDenseBLAS<Double>(shape: [2, 3], elements: [1.0, -1.0, 2.0, 0.0, -2.0, 3.0])
    let vector = TensorDenseBLAS<Double>(shape: [3], elements: [3.0, -1.0, 2.0])
    let product = matrix.times(vector, contract: [(left: 1, right: 0)])

    #expect(product.shape == [2])
    #expect(product.elements == [-3.0, 3.0])
}

@Test func TensorDenseBLAS_matrixMatrixContraction() {
    let left = TensorDenseBLAS<Double>(shape: [2, 3], elements: [1.0, -1.0, 2.0, 0.0, -2.0, 3.0])
    let right = TensorDenseBLAS<Double>(shape: [3, 2], elements: [1.0, 0.0, -1.0, 2.0, 3.0, -2.0])
    let product = left.times(right, contract: [(left: 1, right: 0)])

    #expect(product.shape == [2, 2])
    #expect(product.elements == [3.0, -4.0, 12.0, -8.0])
}

@Test func TensorDenseBLAS_rank3Rank3Contraction() {
    let left = tensor([
        [[1.0, -1.0], [2.0, 0.0]],
        [[0.0, 2.0], [-2.0, 1.0]],
        [[3.0, 1.0], [1.0, -3.0]]
    ])
    let right = tensor([
        [[1.0, 2.0], [-1.0, 0.0], [3.0, -2.0]],
        [[0.0, -1.0], [2.0, 1.0], [-3.0, 1.0]]
    ])
    let product = left.times(right, contract: [(left: 0, right: 1), (left: 2, right: 2)])
    let expected = [[6.0, -5.0], [13.0, -9.0]]

    #expect(product.shape == [2, 2])
    #expect(product.elements == [6.0, 13.0, -5.0, -9.0])
    #expect(product[0, 0] == expected[0][0])
    #expect(product[0, 1] == expected[0][1])
    #expect(product[1, 0] == expected[1][0])
    #expect(product[1, 1] == expected[1][1])
}
