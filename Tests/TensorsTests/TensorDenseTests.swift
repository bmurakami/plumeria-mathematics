import Testing
@testable import Tensors

protocol TensorDenseTestImplementation: TensorArithmetic, TensorMultiplication where S == Double {
    init(shape: [Int], initialValue: Double)
    init(shape: [Int], elements: [Double])
    subscript(_ indices: TensorSliceIndex...) -> Self { get set }
    func flatten() -> [Double]
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

    func checkNestedArrayInitializer() {
        switch self {
        case .reference: verifyNestedArrayInitializer(TensorDenseReference<Double>.self)
        case .blas: verifyNestedArrayInitializer(TensorDenseBLAS<Double>.self)
        }
    }

    func checkFlatArrayRoundTrip() {
        switch self {
        case .reference: verifyFlatArrayRoundTrip(TensorDenseReference<Double>.self)
        case .blas: verifyFlatArrayRoundTrip(TensorDenseBLAS<Double>.self)
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

    func checkIndexMultiply() {
        switch self {
        case .reference: verifyIndexMultiply(TensorDenseReference<Double>.self)
        case .blas: verifyIndexMultiply(TensorDenseBLAS<Double>.self)
        }
    }

    func checkStringMultiply() {
        switch self {
        case .reference: verifyStringMultiply(TensorDenseReference<Double>.self)
        case .blas: verifyStringMultiply(TensorDenseBLAS<Double>.self)
        }
    }

    func checkOuterMultiply() {
        switch self {
        case .reference: verifyOuterMultiply(TensorDenseReference<Double>.self)
        case .blas: verifyOuterMultiply(TensorDenseBLAS<Double>.self)
        }
    }

    func checkPermute() {
        switch self {
        case .reference: verifyPermute(TensorDenseReference<Double>.self)
        case .blas: verifyPermute(TensorDenseBLAS<Double>.self)
        }
    }

    func checkSliceAssignment() {
        switch self {
        case .reference: verifySliceAssignment(TensorDenseReference<Double>.self)
        case .blas: verifySliceAssignment(TensorDenseBLAS<Double>.self)
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
func TensorDense_nestedArrayInitializer(implementation: TensorImplementation) {
    implementation.checkNestedArrayInitializer()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_flatArrayRoundTrip(implementation: TensorImplementation) {
    implementation.checkFlatArrayRoundTrip()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_supportsRankZeroTensors(implementation: TensorImplementation) { implementation.checkRankZeroTensors() }

@Test(arguments: TensorImplementation.allCases)
func TensorDense_elementwiseArithmetic(implementation: TensorImplementation) {
    implementation.checkElementwiseArithmetic()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_indexMultiply(implementation: TensorImplementation) {
    implementation.checkIndexMultiply()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_stringMultiply(implementation: TensorImplementation) {
    implementation.checkStringMultiply()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_outerMultiply(implementation: TensorImplementation) {
    implementation.checkOuterMultiply()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_permute(implementation: TensorImplementation) {
    implementation.checkPermute()
}

@Test(arguments: TensorImplementation.allCases)
func TensorDense_sliceAssignment(implementation: TensorImplementation) {
    implementation.checkSliceAssignment()
}

@Test func TensorNotation_reportsInvalidMultiplyNotation() {
    let left = TensorDenseBLAS<Double>(shape: [2, 3], initialValue: 0.0)
    let right = TensorDenseBLAS<Double>(shape: [3, 2], initialValue: 0.0)

    #expect(tensorMultiplicationNotationValidationError(left, right, "ij -> jk") ==
            "Tensor multiplication notation must not include an output clause")
    #expect(tensorMultiplicationNotationValidationError(left, right, "ij") ==
            "Tensor multiplication notation must contain two operands")
    #expect(tensorMultiplicationNotationValidationError(left, right, "ij,j1") ==
            "Right tensor index '1' must be an ASCII letter")
    #expect(tensorMultiplicationNotationValidationError(left, right, "ij,jk,kl") ==
            "Tensor multiplication notation must contain two operands")
}

@Test func TensorNotation_reportsInvalidMultiplyIndices() {
    let left = TensorDenseBLAS<Double>(shape: [2, 3], initialValue: 0.0)
    let right = TensorDenseBLAS<Double>(shape: [4, 2], initialValue: 0.0)

    #expect(tensorMultiplicationNotationValidationError(left, right, "i,jk") ==
            "Left index count must match tensor rank: got 1, expected 2")
    #expect(tensorMultiplicationNotationValidationError(left, right, "ij,k") ==
            "Right index count must match tensor rank: got 1, expected 2")
    #expect(tensorMultiplicationNotationValidationError(left, right, "ii,jk") == "Left index 'i' must not repeat")
    #expect(tensorMultiplicationNotationValidationError(left, right, "ij,kk") == "Right index 'k' must not repeat")
    #expect(tensorMultiplicationNotationValidationError(left, right, "ij,jk") ==
            "Contracted dimensions for index 'j' must match: left 3, right 4")
}

@Test func TensorNotation_reportsInvalidPermuteNotation() {
    let tensor = TensorDenseBLAS<Double>(shape: [2, 3, 2], initialValue: 0.0)

    #expect(tensorPermutationNotationValidationError(tensor, "ijk") ==
            "Tensor permutation notation must contain an output clause")
    #expect(tensorPermutationNotationValidationError(tensor, "ijk,jik,ikj") ==
            "Tensor permutation notation must contain one tensor")
    #expect(tensorPermutationNotationValidationError(tensor, "ijk -> jik -> kij") ==
            "Tensor permutation notation must contain one output clause")
    #expect(tensorPermutationNotationValidationError(tensor, "ijk -> j1k") ==
            "Destination tensor index '1' must be an ASCII letter")
}

@Test func TensorNotation_reportsInvalidPermuteIndices() {
    let tensor = TensorDenseBLAS<Double>(shape: [2, 3, 2], initialValue: 0.0)

    #expect(tensorPermutationNotationValidationError(tensor, "ij -> ji") ==
            "Source index count must match tensor rank: got 2, expected 3")
    #expect(tensorPermutationNotationValidationError(tensor, "ijk -> ji") ==
            "Destination index count must match tensor rank: got 2, expected 3")
    #expect(tensorPermutationNotationValidationError(tensor, "iik -> jik") == "Source index 'i' must not repeat")
    #expect(tensorPermutationNotationValidationError(tensor, "ijk -> jjk") ==
            "Destination index 'j' must not repeat")
    #expect(tensorPermutationNotationValidationError(tensor, "ijk -> ikl") ==
            "Destination indices must permute source indices")
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

private func verifyNestedArrayInitializer<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let values: TensorNestedArray<Double> = [
        [[1.0, -1.0], [2.0, 0.0]],
        [[0.0, 2.0], [-2.0, 1.0]],
        [[3.0, 1.0], [1.0, -3.0]]
    ]
    let tensor = T(values)

    #expect(tensor.shape == [3, 2, 2])
    #expect(tensor.rank == 3)
    #expect(tensor[[0, 0, 0]] == 1.0)
    #expect(tensor[[1, 1, 0]] == -2.0)
    #expect(tensor[[2, 1, 1]] == -3.0)
}

private func verifyFlatArrayRoundTrip<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let elements = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0, 1.0, -2.0, 2.0, 3.0, 0.0]
    let tensor = T(shape: [2, 3, 2], elements: elements)
    let copy = T(shape: tensor.shape, elements: tensor.flatten())

    #expect(tensor.shape == [2, 3, 2])
    #expect(tensor[[0, 0, 0]] == 1.0)
    #expect(tensor[[1, 0, 0]] == -1.0)
    #expect(tensor[[0, 1, 0]] == 2.0)
    #expect(tensor[[1, 2, 1]] == 0.0)
    #expect(tensor.flatten() == elements)
    #expect(copy == tensor)
}

private func verifySliceAssignment<T: TensorDenseTestImplementation>(_ type: T.Type) {
    var tensor = T(shape: [2, 3, 2], elements: [
        1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0, 1.0, -2.0, 2.0, 3.0, 0.0
    ])
    tensor[all, range(1..<3), 0] = T(shape: [2, 2], elements: [20.0, 30.0, 40.0, 50.0])

    #expect(tensor[[0, 1, 0]] == 20.0)
    #expect(tensor[[1, 1, 0]] == 30.0)
    #expect(tensor[[0, 2, 0]] == 40.0)
    #expect(tensor[[1, 2, 0]] == 50.0)
    #expect(tensor[[0, 1, 1]] == -2.0)
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

private func verifyIndexMultiply<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let i = TensorIndex("i"), j = TensorIndex("j"), k = TensorIndex("k")
    let leftValues: TensorNestedArray<Double> = [
        [1.0, 2.0, 0.0],
        [-1.0, 3.0, 1.0]
    ]
    let rightValues: TensorNestedArray<Double> = [
        [2.0, 1.0],
        [0.0, -1.0],
        [3.0, 2.0]
    ]
    let product = multiply(T(leftValues), [i, j], T(rightValues), [j, k])

    #expect(product.shape == [2, 2])
    #expect(product[[0, 0]] == 2.0)
    #expect(product[[1, 0]] == 1.0)
    #expect(product[[0, 1]] == -1.0)
    #expect(product[[1, 1]] == -2.0)
}

private func verifyPermute<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let i = TensorIndex("i"), j = TensorIndex("j"), k = TensorIndex("k")
    let values: TensorNestedArray<Double> = [
        [[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]],
        [[0.0, 2.0], [-2.0, 1.0], [1.0, -3.0]]
    ]
    let indexed = permute(T(values), from: [i, j, k], to: [j, i, k])
    let string = permute(T(values), "ijk -> jik")

    #expect(indexed.shape == [3, 2, 2])
    #expect(string.shape == indexed.shape)
    #expect(indexed[[1, 0, 0]] == 2.0)
    #expect(indexed[[2, 1, 1]] == -3.0)
    #expect(string == indexed)
}

private func verifyStringMultiply<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let leftValues: TensorNestedArray<Double> = [
        [1.0, 2.0, 0.0],
        [-1.0, 3.0, 1.0]
    ]
    let rightValues: TensorNestedArray<Double> = [
        [2.0, 1.0],
        [0.0, -1.0],
        [3.0, 2.0]
    ]
    let product = multiply(T(leftValues), T(rightValues), "ij, jk")

    #expect(product.shape == [2, 2])
    #expect(product[[0, 0]] == 2.0)
    #expect(product[[1, 0]] == 1.0)
    #expect(product[[0, 1]] == -1.0)
    #expect(product[[1, 1]] == -2.0)
}

private func verifyOuterMultiply<T: TensorDenseTestImplementation>(_ type: T.Type) {
    let i = TensorIndex("i"), j = TensorIndex("j"), k = TensorIndex("k"), l = TensorIndex("l")
    let leftValues: TensorNestedArray<Double> = [[1.0, 2.0], [-1.0, 0.0]]
    let rightValues: TensorNestedArray<Double> = [[2.0, 1.0], [0.0, -1.0], [3.0, 2.0]]
    let indexed = multiply(T(leftValues), [i, j], T(rightValues), [k, l])
    let string = multiply(T(leftValues), T(rightValues), "ij, kl")

    #expect(indexed.shape == [2, 2, 3, 2])
    #expect(string.shape == indexed.shape)
    #expect(indexed[[0, 0, 0, 0]] == 2.0)
    #expect(indexed[[1, 0, 2, 1]] == -2.0)
    #expect(indexed[[0, 1, 1, 1]] == -2.0)
    #expect(string == indexed)
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
