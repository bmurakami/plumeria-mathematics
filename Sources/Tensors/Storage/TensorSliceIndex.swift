public enum TensorSliceIndex: Equatable, Sendable {
    case index(Int)
    case range(Range<Int>)
    case step(Range<Int>, by: Int)
    case all

    func sliceRange(dimensionSize: Int) -> SliceRange {
        precondition(dimensionSize >= 0, "Dimension size must be non-negative")
        switch self {
        case .index:
            preconditionFailure("Integer index cannot be used as a slice range")
        case .range(let range):
            return SliceRange(range)
        case .step(let range, let step):
            return SliceRange(range, step: step)
        case .all:
            return SliceRange.all(length: dimensionSize)
        }
    }

    func fixedIndex(dimensionSize: Int) -> Int {
        precondition(dimensionSize >= 0, "Dimension size must be non-negative")
        switch self {
        case .index(let index):
            precondition(index >= 0 && index < dimensionSize, "Tensor index out of bounds")
            return index
        case .range, .step, .all:
            preconditionFailure("Slice index cannot be used as a fixed integer index")
        }
    }
}

public let all = TensorSliceIndex.all

public func range(_ range: Range<Int>) -> TensorSliceIndex {
    .range(range)
}

public func step(_ range: Range<Int>, by: Int) -> TensorSliceIndex {
    .step(range, by: by)
}

// MARK: - Integer Literals

extension TensorSliceIndex: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self = .index(value)
    }
}
