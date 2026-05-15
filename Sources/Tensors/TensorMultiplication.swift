public protocol TensorMultiplication: TensorStructure {
    associatedtype S: PluScalar
    associatedtype MatrixImplementation: PluMatrix where MatrixImplementation.S == S

    init(shape: [Int], initialValue: S)
    subscript(_ indices: [Int]) -> S { get set }
}

extension TensorMultiplication {
    public func times<T: TensorMultiplication>(_ other: T, contract axes: [(left: Int, right: Int)]) -> Self
    where T.S == S {
        validateContraction(axes, with: other)
        let leftContractedAxes = Set(axes.map(\.left))
        let rightContractedAxes = Set(axes.map(\.right))
        let leftFreeAxes = (0..<rank).filter { !leftContractedAxes.contains($0) }
        let rightFreeAxes = (0..<other.rank).filter { !rightContractedAxes.contains($0) }
        let leftFreeShape = leftFreeAxes.map { shape[$0] }
        let rightFreeShape = rightFreeAxes.map { other.shape[$0] }
        let contractShape = axes.map { shape[$0.left] }
        let resultShape = leftFreeShape + rightFreeShape
        var result = Self(shape: resultShape, initialValue: .zero)
        let leftRows = Self.countElements(for: leftFreeShape)
        let shared = Self.countElements(for: contractShape)
        let rightColumns = Self.countElements(for: rightFreeShape)
        if Self.countElements(for: resultShape) == 0 || shared == 0 { return result }
        let leftMatrix = matricizedLeftTensor(freeAxes: leftFreeAxes, contractAxes: axes.map(\.left))
        let rightMatrix = other.matricizedRightTensor(freeAxes: rightFreeAxes, contractAxes: axes.map(\.right))
        let product = leftMatrix * rightMatrix
        for resultIndex in Self.indexCombinations(for: resultShape) {
            let leftFreeIndex = Array(resultIndex.prefix(leftFreeAxes.count))
            let rightFreeIndex = Array(resultIndex.dropFirst(leftFreeAxes.count))
            let row = Self.linearIndex(leftFreeIndex, shape: leftFreeShape)
            let column = Self.linearIndex(rightFreeIndex, shape: rightFreeShape)
            precondition(row < leftRows && column < rightColumns, "Tensor contraction output index is out of bounds")
            result[resultIndex] = product[row, column]
        }
        return result
    }

    private func matricizedLeftTensor(freeAxes: [Int], contractAxes: [Int]) -> MatrixImplementation {
        let freeShape = freeAxes.map { shape[$0] }
        let contractShape = contractAxes.map { shape[$0] }
        var matrix = MatrixImplementation(
            rows: Self.countElements(for: freeShape),
            columns: Self.countElements(for: contractShape),
            initialValue: .zero
        )
        for freeIndex in Self.indexCombinations(for: freeShape) {
            for contractIndex in Self.indexCombinations(for: contractShape) {
                var tensorIndex = Array(repeating: 0, count: rank)
                for (position, axis) in freeAxes.enumerated() { tensorIndex[axis] = freeIndex[position] }
                for (position, axis) in contractAxes.enumerated() { tensorIndex[axis] = contractIndex[position] }
                matrix[
                    Self.linearIndex(freeIndex, shape: freeShape),
                    Self.linearIndex(contractIndex, shape: contractShape)
                ] = self[tensorIndex]
            }
        }
        return matrix
    }

    private func matricizedRightTensor(freeAxes: [Int], contractAxes: [Int]) -> MatrixImplementation {
        let freeShape = freeAxes.map { shape[$0] }
        let contractShape = contractAxes.map { shape[$0] }
        var matrix = MatrixImplementation(
            rows: Self.countElements(for: contractShape),
            columns: Self.countElements(for: freeShape),
            initialValue: .zero
        )
        for contractIndex in Self.indexCombinations(for: contractShape) {
            for freeIndex in Self.indexCombinations(for: freeShape) {
                var tensorIndex = Array(repeating: 0, count: rank)
                for (position, axis) in contractAxes.enumerated() { tensorIndex[axis] = contractIndex[position] }
                for (position, axis) in freeAxes.enumerated() { tensorIndex[axis] = freeIndex[position] }
                matrix[
                    Self.linearIndex(contractIndex, shape: contractShape),
                    Self.linearIndex(freeIndex, shape: freeShape)
                ] = self[tensorIndex]
            }
        }
        return matrix
    }

    private func validateContraction<T: TensorMultiplication>(_ axes: [(left: Int, right: Int)], with other: T)
    where T.S == S {
        var leftAxes = Set<Int>()
        var rightAxes = Set<Int>()
        for axisPair in axes {
            precondition(axisPair.left >= 0 && axisPair.left < rank, "Left contraction axis is out of bounds")
            precondition(axisPair.right >= 0 && axisPair.right < other.rank, "Right contraction axis is out of bounds")
            precondition(leftAxes.insert(axisPair.left).inserted, "Left contraction axes must be unique")
            precondition(rightAxes.insert(axisPair.right).inserted, "Right contraction axes must be unique")
            precondition(shape[axisPair.left] == other.shape[axisPair.right], "Contracted dimensions must match")
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
