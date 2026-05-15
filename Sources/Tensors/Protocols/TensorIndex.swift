public struct TensorIndex: Hashable, ExpressibleByStringLiteral {
    public let symbol: String

    public init(_ symbol: String) {
        precondition(!symbol.isEmpty, "Tensor index symbol must not be empty"); self.symbol = symbol
    }
    public init(stringLiteral value: String) { self.init(value) }
}

public func multiply<L: TensorMultiplication, R: TensorMultiplication>(
    _ left: L,
    _ leftIndices: [TensorIndex],
    _ right: R,
    _ rightIndices: [TensorIndex]
) -> L where L.S == R.S {
    precondition(leftIndices.count == left.rank, "Left index count must match tensor rank")
    precondition(rightIndices.count == right.rank, "Right index count must match tensor rank")
    precondition(Set(leftIndices).count == leftIndices.count, "Left indices must not repeat")
    precondition(Set(rightIndices).count == rightIndices.count, "Right indices must not repeat")
    let rightAxisByIndex = Dictionary(uniqueKeysWithValues: rightIndices.enumerated().map { ($0.element, $0.offset) })
    let axes = leftIndices.enumerated().compactMap { leftAxis, index -> (left: Int, right: Int)? in
        guard let rightAxis = rightAxisByIndex[index] else { return nil }
        return (leftAxis, rightAxis)
    }
    return left.times(right, contract: axes)
}

public func multiply<L: TensorMultiplication, R: TensorMultiplication>(
    _ left: L,
    _ right: R,
    _ notation: String
) -> L where L.S == R.S {
    let compact = notation.filter { !$0.isWhitespace }
    precondition(!compact.contains("->"), "Tensor multiplication notation must not include an output clause")
    let operands = compact.split(separator: ",", omittingEmptySubsequences: false)
    precondition(operands.count == 2, "Tensor multiplication notation must contain two operands")
    return multiply(left, tensorIndices(operands[0]), right, tensorIndices(operands[1]))
}

private func tensorIndices(_ symbols: Substring) -> [TensorIndex] { symbols.map { TensorIndex(String($0)) } }

public func permute<T: TensorMultiplication>(
    _ tensor: T,
    from sourceIndices: [TensorIndex],
    to destinationIndices: [TensorIndex]
) -> T {
    precondition(sourceIndices.count == tensor.rank, "Source index count must match tensor rank")
    precondition(destinationIndices.count == tensor.rank, "Destination index count must match tensor rank")
    precondition(Set(sourceIndices).count == sourceIndices.count, "Source indices must not repeat")
    precondition(Set(destinationIndices) == Set(sourceIndices), "Destination indices must permute source indices")
    let sourceAxisByIndex = Dictionary(uniqueKeysWithValues: sourceIndices.enumerated().map { ($0.element, $0.offset) })
    let sourceAxes = destinationIndices.map { sourceAxisByIndex[$0]! }
    let resultShape = sourceAxes.map { tensor.shape[$0] }
    var result = T(shape: resultShape, initialValue: .zero)
    for resultIndex in indexCombinations(for: resultShape) {
        var sourceIndex = Array(repeating: 0, count: tensor.rank)
        for (resultAxis, sourceAxis) in sourceAxes.enumerated() { sourceIndex[sourceAxis] = resultIndex[resultAxis] }
        result[resultIndex] = tensor[sourceIndex]
    }
    return result
}

public func permute<T: TensorMultiplication>(_ tensor: T, _ notation: String) -> T {
    let compact = notation.filter { !$0.isWhitespace }
    guard let arrow = compact.range(of: "->") else {
        preconditionFailure("Tensor permutation notation must contain an output clause")
    }
    let source = compact[..<arrow.lowerBound]
    let destination = compact[arrow.upperBound...]
    precondition(!destination.contains("->"), "Tensor permutation notation must contain one output clause")
    return permute(tensor, from: tensorIndices(source), to: tensorIndices(destination))
}

private func indexCombinations(for shape: [Int]) -> [[Int]] {
    if shape.isEmpty { return [[]] }
    if shape.contains(0) { return [] }
    return (0..<shape.reduce(1, *)).map { flatIndex in
        var remaining = flatIndex
        return shape.map { dimension in
            let index = remaining % dimension
            remaining /= dimension
            return index
        }
    }
}
