public protocol MatrixColumnMajorInitializable: PluMatrix {
    init(rows: Int, columns: Int, values: [S])
}

public protocol TensorMultiplication: TensorStructure {
    associatedtype MatrixImplementation: MatrixColumnMajorInitializable where MatrixImplementation.S == S

    var elements: [S] { get }
    init(shape: [Int], initialValue: S)
    subscript(_ indices: [Int]) -> S { get set }
}

extension TensorMultiplication {
    public func times<T: TensorMultiplication>(_ other: T, contract indices: [(left: Int, right: Int)]) -> Self
    where T.S == S {
        validateContraction(indices, with: other)
        let leftContractedIndices = Set(indices.map(\.left))
        let rightContractedIndices = Set(indices.map(\.right))
        let leftFreeIndices = (0..<rank).filter { !leftContractedIndices.contains($0) }
        let rightFreeIndices = (0..<other.rank).filter { !rightContractedIndices.contains($0) }
        let leftFreeShape = leftFreeIndices.map { shape[$0] }
        let rightFreeShape = rightFreeIndices.map { other.shape[$0] }
        let contractShape = indices.map { shape[$0.left] }
        let resultShape = leftFreeShape + rightFreeShape
        var result = Self(shape: resultShape, initialValue: .zero)
        let leftRows = Self.countElements(for: leftFreeShape)
        let shared = Self.countElements(for: contractShape)
        let rightColumns = Self.countElements(for: rightFreeShape)
        if Self.countElements(for: resultShape) == 0 || shared == 0 { return result }
        let leftMatrix = matricizedLeftTensor(freeIndices: leftFreeIndices, contractIndices: indices.map(\.left))
        let rightMatrix = other.matricizedRightTensor(
            freeIndices: rightFreeIndices,
            contractIndices: indices.map(\.right)
        )
        let product = leftMatrix * rightMatrix
        for resultIndex in Self.indexCombinations(for: resultShape) {
            let leftFreeIndex = Array(resultIndex.prefix(leftFreeIndices.count))
            let rightFreeIndex = Array(resultIndex.dropFirst(leftFreeIndices.count))
            let row = Self.linearIndex(leftFreeIndex, shape: leftFreeShape)
            let column = Self.linearIndex(rightFreeIndex, shape: rightFreeShape)
            precondition(row < leftRows && column < rightColumns, "Tensor contraction output index is out of bounds")
            result[resultIndex] = product[row, column]
        }
        return result
    }

    private func matricizedLeftTensor(freeIndices: [Int], contractIndices: [Int]) -> MatrixImplementation {
        let freeShape = freeIndices.map { shape[$0] }
        let contractShape = contractIndices.map { shape[$0] }
        if freeIndices + contractIndices == Array(0..<rank) {
            return MatrixImplementation(
                rows: Self.countElements(for: freeShape),
                columns: Self.countElements(for: contractShape),
                values: elements
            )
        }
        var matrix = MatrixImplementation(
            rows: Self.countElements(for: freeShape),
            columns: Self.countElements(for: contractShape),
            initialValue: .zero
        )
        for freeIndex in Self.indexCombinations(for: freeShape) {
            for contractIndex in Self.indexCombinations(for: contractShape) {
                var tensorIndex = Array(repeating: 0, count: rank)
                for (position, index) in freeIndices.enumerated() { tensorIndex[index] = freeIndex[position] }
                for (position, index) in contractIndices.enumerated() { tensorIndex[index] = contractIndex[position] }
                matrix[
                    Self.linearIndex(freeIndex, shape: freeShape),
                    Self.linearIndex(contractIndex, shape: contractShape)
                ] = self[tensorIndex]
            }
        }
        return matrix
    }

    private func matricizedRightTensor(freeIndices: [Int], contractIndices: [Int]) -> MatrixImplementation {
        let freeShape = freeIndices.map { shape[$0] }
        let contractShape = contractIndices.map { shape[$0] }
        if contractIndices + freeIndices == Array(0..<rank) {
            return MatrixImplementation(
                rows: Self.countElements(for: contractShape),
                columns: Self.countElements(for: freeShape),
                values: elements
            )
        }
        var matrix = MatrixImplementation(
            rows: Self.countElements(for: contractShape),
            columns: Self.countElements(for: freeShape),
            initialValue: .zero
        )
        for contractIndex in Self.indexCombinations(for: contractShape) {
            for freeIndex in Self.indexCombinations(for: freeShape) {
                var tensorIndex = Array(repeating: 0, count: rank)
                for (position, index) in contractIndices.enumerated() { tensorIndex[index] = contractIndex[position] }
                for (position, index) in freeIndices.enumerated() { tensorIndex[index] = freeIndex[position] }
                matrix[
                    Self.linearIndex(contractIndex, shape: contractShape),
                    Self.linearIndex(freeIndex, shape: freeShape)
                ] = self[tensorIndex]
            }
        }
        return matrix
    }

    private func validateContraction<T: TensorMultiplication>(_ indices: [(left: Int, right: Int)], with other: T)
    where T.S == S {
        var leftIndices = Set<Int>()
        var rightIndices = Set<Int>()
        for indexPair in indices {
            precondition(indexPair.left >= 0 && indexPair.left < rank, "Left contraction index is out of bounds")
            precondition(
                indexPair.right >= 0 && indexPair.right < other.rank,
                "Right contraction index is out of bounds"
            )
            precondition(leftIndices.insert(indexPair.left).inserted, "Left contraction indices must be unique")
            precondition(rightIndices.insert(indexPair.right).inserted, "Right contraction indices must be unique")
            precondition(shape[indexPair.left] == other.shape[indexPair.right], "Contracted dimensions must match")
        }
    }

    private static func indexCombinations(for shape: [Int]) -> [[Int]] {
        if shape.isEmpty { return [[]] }
        if shape.contains(0) { return [] }
        return (0..<countElements(for: shape)).map { flatIndex in
            var remaining = flatIndex
            return shape.map { dimension in
                let index = remaining % dimension
                remaining /= dimension
                return index
            }
        }
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

    private static func countElements(for shape: [Int]) -> Int { shape.reduce(1, *) }
}
