import Testing
@testable import Tensors

private struct TestTensor<S: PluScalar>: TensorMultiplication {
    typealias MatrixImplementation = MatrixDenseReference<S>

    var shape: [Int]
    var rank: Int { shape.count }
    var elements: [S]

    init(shape: [Int], initialValue: S) {
        self.shape = shape
        self.elements = Array(repeating: initialValue, count: shape.reduce(1, *))
    }

    init(shape: [Int], elements: [S]) {
        self.shape = shape
        self.elements = elements
    }

    init(_ values: TensorNestedArray<S>) {
        self.shape = values.shape
        self.elements = values.flatten()
    }

    subscript(_ indices: [Int]) -> S {
        get { elements[Self.linearIndex(indices, shape: shape)] }
        set { elements[Self.linearIndex(indices, shape: shape)] = newValue }
    }

    private static func linearIndex(_ indices: [Int], shape: [Int]) -> Int {
        var stride = 1
        var flatIndex = 0
        for (index, dimension) in zip(indices, shape) {
            flatIndex += index * stride
            stride *= dimension
        }
        return flatIndex
    }
}

@Test func TensorMultiplication_outerProduct() {
    let left = TestTensor<Double>(shape: [2], elements: [2.0, 3.0])
    let right = TestTensor<Double>(shape: [3], elements: [5.0, 7.0, 11.0])
    let product = left.times(right, contract: [])

    #expect(product.shape == [2, 3])
    #expect(product.elements == [10.0, 15.0, 14.0, 21.0, 22.0, 33.0])
}

@Test func TensorMultiplication_dotProductReturnsRankZeroTensor() {
    let left = TestTensor<Double>(shape: [3], elements: [2.0, 3.0, 5.0])
    let right = TestTensor<Double>(shape: [3], elements: [7.0, 11.0, 13.0])
    let product = left.times(right, contract: [(left: 0, right: 0)])

    #expect(product.shape == [])
    #expect(product.rank == 0)
    #expect(product.elements == [112.0])
    #expect(product[[]] == 112.0)
}

@Test func TensorMultiplication_matrixVectorContraction() {
    let matrix = TestTensor<Double>(shape: [2, 3], elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    let vector = TestTensor<Double>(shape: [3], elements: [7.0, 11.0, 13.0])
    let product = matrix.times(vector, contract: [(left: 1, right: 0)])

    #expect(product.shape == [2])
    #expect(product.elements == [68.0, 161.0])
}

@Test func TensorMultiplication_matrixMatrixContraction() {
    let left = TestTensor<Double>(shape: [2, 3], elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    let right = TestTensor<Double>(shape: [3, 2], elements: [7.0, 9.0, 11.0, 8.0, 10.0, 12.0])
    let product = left.times(right, contract: [(left: 1, right: 0)])

    #expect(product.shape == [2, 2])
    #expect(product.elements == [58.0, 139.0, 64.0, 154.0])
}

@Test func TensorMultiplication_multiAxisContraction() {
    let left = TestTensor<Double>(shape: [2, 3, 4], elements: Array(1...24).map(Double.init))
    let right = TestTensor<Double>(shape: [3, 5, 4], elements: Array(1...60).map(Double.init))
    let product = left.times(right, contract: [(left: 1, right: 0), (left: 2, right: 2)])

    #expect(product.shape == [2, 5])
    #expect(product.elements == [
        4894.0, 5188.0, 5326.0, 5656.0, 5758.0,
        6124.0, 6190.0, 6592.0, 6622.0, 7060.0
    ])
}
