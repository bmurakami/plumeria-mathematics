import Testing
@testable import Tensors

protocol TensorDenseTestImplementation: TensorArithmetic, TensorStructure where S == Double, Magnitude == Double {
    init(shape: [Int], initialValue: Double)
    subscript(_ indices: [Int]) -> Double { get set }
}

extension TensorDenseReference: TensorDenseTestImplementation where S == Double {}
extension TensorDenseBLAS: TensorDenseTestImplementation where S == Double {}

enum TensorImplementation: CaseIterable, CustomStringConvertible {
    case reference
    case blas

    var description: String {
        switch self {
        case .reference: "reference"
        case .blas: "blas"
        }
    }

    func checkInitializesWithValue() {
        switch self {
        case .reference: verifyInitializesWithValue(TensorDenseReference<Double>.self)
        case .blas: verifyInitializesWithValue(TensorDenseBLAS<Double>.self)
        }
    }

    func checkReadsAndMutatesElements() {
        switch self {
        case .reference: verifyReadsAndMutatesElements(TensorDenseReference<Double>.self)
        case .blas: verifyReadsAndMutatesElements(TensorDenseBLAS<Double>.self)
        }
    }

    func checkRankZeroTensors() {
        switch self {
        case .reference: verifyRankZeroTensors(TensorDenseReference<Double>.self)
        case .blas: verifyRankZeroTensors(TensorDenseBLAS<Double>.self)
        }
    }

    func checkElementwiseArithmetic() {
        switch self {
        case .reference: verifyElementwiseArithmetic(TensorDenseReference<Double>.self)
        case .blas: verifyElementwiseArithmetic(TensorDenseBLAS<Double>.self)
        }
    }
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_initializesWithValue(implementation: TensorImplementation) {
    implementation.checkInitializesWithValue()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_readsAndMutatesElements(implementation: TensorImplementation) {
    implementation.checkReadsAndMutatesElements()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_supportsRankZeroTensors(implementation: TensorImplementation) {
    implementation.checkRankZeroTensors()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_elementwiseArithmetic(implementation: TensorImplementation) {
    implementation.checkElementwiseArithmetic()
}

private func verifyInitializesWithValue<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let tensor = T(shape: [2, 3], initialValue: 2.0)

    #expect(tensor.shape == [2, 3])
    #expect(tensor.rank == 2)
    #expect(tensor[[0, 0]] == 2.0)
    #expect(tensor[[1, 2]] == 2.0)
}

private func verifyReadsAndMutatesElements<T: TensorDenseTestImplementation>(_ type: T.Type) {
    var tensor = T(shape: [2, 3], initialValue: 0.0)
    tensor[[0, 0]] = 1.0
    tensor[[1, 0]] = -1.0
    tensor[[0, 2]] = 3.0

    #expect(tensor[[0, 0]] == 1.0)
    #expect(tensor[[1, 0]] == -1.0)
    #expect(tensor[[0, 2]] == 3.0)
}

private func verifyRankZeroTensors<T: TensorDenseTestImplementation>(_ type: T.Type) {
    var tensor = T(shape: [], initialValue: 7.0)

    #expect(tensor.shape == [])
    #expect(tensor.rank == 0)
    #expect(tensor[[]] == 7.0)

    tensor[[]] = -2.0
    #expect(tensor[[]] == -2.0)
}

private func verifyElementwiseArithmetic<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let left = rank3Tensor(T.self, values: [
        [[1.0, -1.0], [2.0, 0.0]],
        [[0.0, 2.0], [-2.0, 1.0]]
    ])
    let right = rank3Tensor(T.self, values: [
        [[2.0, 1.0], [-1.0, 3.0]],
        [[-3.0, 0.0], [1.0, -2.0]]
    ])
    let sum = left + right
    let negative = -left
    let scaledRight = left * 2.0
    let scaledLeft = 2.0 * left
    let divided = left / 2.0

    #expect(sum[[0, 0, 0]] == 3.0)
    #expect(sum[[1, 1, 1]] == -1.0)
    #expect(negative[[0, 1, 0]] == -2.0)
    #expect(scaledRight[[1, 1, 0]] == -4.0)
    #expect(scaledLeft == scaledRight)
    #expect(divided[[0, 0, 1]] == -0.5)
    #expect(sum.isApproximatelyEqual(to: sum, relativeTolerance: 1e-12, norm: { _ in 0.0 }))
}

private func rank3Tensor<T: TensorDenseTestImplementation>(_ type: T.Type, values: [[[Double]]]) -> T {
    let shape = [values.count, values[0].count, values[0][0].count]
    var tensor = T(shape: shape, initialValue: 0.0)
    for i in 0..<shape[0] {
        for j in 0..<shape[1] {
            for k in 0..<shape[2] { tensor[[i, j, k]] = values[i][j][k] }
        }
    }
    return tensor
}
