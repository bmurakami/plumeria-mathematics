public struct TensorIndex: Hashable, ExpressibleByStringLiteral {
    public let symbol: String

    public init(_ symbol: String) {
        precondition(!symbol.isEmpty, "Tensor index symbol must not be empty")
        self.symbol = symbol
    }

    public init(stringLiteral value: String) { self.init(value) }
}

struct TensorNotationError: Error {
    let message: String
}

public func multiply<L: TensorMultiplication, R: TensorMultiplication>(
    _ left: L, _ leftIndices: [TensorIndex], _ right: R, _ rightIndices: [TensorIndex]
) -> L where L.S == R.S {
    if let error = tensorMultiplicationValidationError(left, leftIndices, right, rightIndices) {
        preconditionFailure(error)
    }
    let rightPositionByIndex = Dictionary(
        uniqueKeysWithValues: rightIndices.enumerated().map { ($0.element, $0.offset) }
    )
    let contractedIndices = leftIndices.enumerated().compactMap { leftPosition, index -> (left: Int, right: Int)? in
        guard let rightPosition = rightPositionByIndex[index] else { return nil }
        return (leftPosition, rightPosition)
    }
    return left.times(right, contract: contractedIndices)
}

public func multiply<L: TensorMultiplication, R: TensorMultiplication>(_ left: L, _ right: R, _ notation: String) -> L
    where L.S == R.S {
    switch tensorMultiplicationIndices(left, right, notation) {
    case let .success(indices): return multiply(left, indices.left, right, indices.right)
    case let .failure(error): preconditionFailure(error.message)
    }
}

public func permute<T: TensorMultiplication>(
    _ tensor: T, from sourceIndices: [TensorIndex], to destinationIndices: [TensorIndex]
) -> T {
    if let error = tensorPermutationValidationError(tensor, from: sourceIndices, to: destinationIndices) {
        preconditionFailure(error)
    }
    let sourcePositionByIndex = Dictionary(
        uniqueKeysWithValues: sourceIndices.enumerated().map { ($0.element, $0.offset) }
    )
    let sourcePositions = destinationIndices.map { sourcePositionByIndex[$0]! }
    let resultShape = sourcePositions.map { tensor.shape[$0] }
    var result = T(shape: resultShape, initialValue: .zero)
    for resultIndex in indexCombinations(for: resultShape) {
        var sourceIndex = Array(repeating: 0, count: tensor.rank)
        for (resultPosition, sourcePosition) in sourcePositions.enumerated() {
            sourceIndex[sourcePosition] = resultIndex[resultPosition]
        }
        result[resultIndex] = tensor[sourceIndex]
    }
    return result
}

public func permute<T: TensorMultiplication>(_ tensor: T, _ notation: String) -> T {
    switch tensorPermutationIndices(tensor, notation) {
    case let .success(indices): return permute(tensor, from: indices.source, to: indices.destination)
    case let .failure(error): preconditionFailure(error.message)
    }
}

func tensorMultiplicationValidationError<L: TensorMultiplication, R: TensorMultiplication>(
    _ left: L, _ leftIndices: [TensorIndex], _ right: R, _ rightIndices: [TensorIndex]
) -> String? where L.S == R.S {
    if leftIndices.count != left.rank {
        return "Left index count must match tensor rank: got \(leftIndices.count), expected \(left.rank)"
    }
    if rightIndices.count != right.rank {
        return "Right index count must match tensor rank: got \(rightIndices.count), expected \(right.rank)"
    }
    if let index = duplicateIndex(in: leftIndices) { return "Left index '\(index.symbol)' must not repeat" }
    if let index = duplicateIndex(in: rightIndices) { return "Right index '\(index.symbol)' must not repeat" }
    let rightPositionByIndex = Dictionary(
        uniqueKeysWithValues: rightIndices.enumerated().map { ($0.element, $0.offset) }
    )
    for (leftPosition, index) in leftIndices.enumerated() {
        guard let rightPosition = rightPositionByIndex[index] else { continue }
        if left.shape[leftPosition] != right.shape[rightPosition] {
            let leftDimension = left.shape[leftPosition], rightDimension = right.shape[rightPosition]
            let details = "left \(leftDimension), right \(rightDimension)"
            return "Contracted dimensions for index '\(index.symbol)' must match: \(details)"
        }
    }
    return nil
}

func tensorMultiplicationNotationValidationError<L: TensorMultiplication, R: TensorMultiplication>(
    _ left: L, _ right: R, _ notation: String
) -> String? where L.S == R.S {
    switch tensorMultiplicationIndices(left, right, notation) {
    case let .success(indices): return tensorMultiplicationValidationError(left, indices.left, right, indices.right)
    case let .failure(error): return error.message
    }
}

func tensorPermutationValidationError<T: TensorMultiplication>(
    _ tensor: T, from sourceIndices: [TensorIndex], to destinationIndices: [TensorIndex]
) -> String? {
    if sourceIndices.count != tensor.rank {
        return "Source index count must match tensor rank: got \(sourceIndices.count), expected \(tensor.rank)"
    }
    if destinationIndices.count != tensor.rank {
        let count = destinationIndices.count
        return "Destination index count must match tensor rank: got \(count), expected \(tensor.rank)"
    }
    if let index = duplicateIndex(in: sourceIndices) { return "Source index '\(index.symbol)' must not repeat" }
    if let index = duplicateIndex(in: destinationIndices) {
        return "Destination index '\(index.symbol)' must not repeat"
    }
    if Set(destinationIndices) != Set(sourceIndices) { return "Destination indices must permute source indices" }
    return nil
}

func tensorPermutationNotationValidationError<T: TensorMultiplication>(_ tensor: T, _ notation: String) -> String? {
    switch tensorPermutationIndices(tensor, notation) {
    case let .success(indices):
        return tensorPermutationValidationError(tensor, from: indices.source, to: indices.destination)
    case let .failure(error): return error.message
    }
}

private func tensorMultiplicationIndices<L: TensorMultiplication, R: TensorMultiplication>(
    _ left: L, _ right: R, _ notation: String
) -> Result<(left: [TensorIndex], right: [TensorIndex]), TensorNotationError> where L.S == R.S {
    let compact = notation.filter { !$0.isWhitespace }
    if compact.contains("->") {
        let error = TensorNotationError(message: "Tensor multiplication notation must not include an output clause")
        return .failure(error)
    }
    let operands = compact.split(separator: ",", omittingEmptySubsequences: false)
    if operands.count != 2 {
        return .failure(TensorNotationError(message: "Tensor multiplication notation must contain two operands"))
    }
    return indexSymbols(operands[0], role: "Left").flatMap { left in
        indexSymbols(operands[1], role: "Right").map { right in (left, right) }
    }
}

private func tensorPermutationIndices<T: TensorMultiplication>(
    _ tensor: T, _ notation: String
) -> Result<(source: [TensorIndex], destination: [TensorIndex]), TensorNotationError> {
    let compact = notation.filter { !$0.isWhitespace }
    if compact.contains(",") {
        return .failure(TensorNotationError(message: "Tensor permutation notation must contain one tensor"))
    }
    guard let arrow = compact.range(of: "->") else {
        return .failure(TensorNotationError(message: "Tensor permutation notation must contain an output clause"))
    }
    let source = compact[..<arrow.lowerBound]
    let destination = compact[arrow.upperBound...]
    if destination.contains("->") {
        return .failure(TensorNotationError(message: "Tensor permutation notation must contain one output clause"))
    }
    return indexSymbols(source, role: "Source").flatMap { source in
        indexSymbols(destination, role: "Destination").map { destination in (source, destination) }
    }
}

private func indexSymbols(_ symbols: Substring, role: String) -> Result<[TensorIndex], TensorNotationError> {
    var indices: [TensorIndex] = []
    indices.reserveCapacity(symbols.count)
    for symbol in symbols {
        guard isIndexNotationSymbol(symbol) else {
            return .failure(TensorNotationError(message: "\(role) tensor index '\(symbol)' must be an ASCII letter"))
        }
        indices.append(TensorIndex(String(symbol)))
    }
    return .success(indices)
}

private func isIndexNotationSymbol(_ symbol: Character) -> Bool {
    guard symbol.unicodeScalars.count == 1, let value = symbol.unicodeScalars.first?.value else { return false }
    return (65...90).contains(value) || (97...122).contains(value)
}

private func duplicateIndex(in indices: [TensorIndex]) -> TensorIndex? {
    var seen = Set<TensorIndex>()
    return indices.first { !seen.insert($0).inserted }
}

private func indexCombinations(for shape: [Int]) -> [[Int]] {
    // Example: indexCombinations(for: [2, 3]) returns [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]].
    if shape.isEmpty { return [[]] }
    if shape.contains(0) { return [] }
    return (0..<shape.reduce(1, *)).map { flatIndex in
        var remaining = flatIndex
        return shape.map { dimension in
            let i = remaining % dimension
            remaining /= dimension
            return i
        }
    }
}
