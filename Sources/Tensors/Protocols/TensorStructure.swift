public enum TensorNestedArray<S: PluScalar>: Equatable {
    case scalar(S)
    indirect case array([TensorNestedArray<S>])

    public var shape: [Int] {
        switch self {
        case .scalar:
            return []
        case .array(let subtensor):
            precondition(!subtensor.isEmpty, "Cannot infer tensor shape from an empty nested array")
            let subtensorShape = subtensor[0].shape
            precondition(
                subtensor.allSatisfy { $0.shape == subtensorShape },
                "Tensor nested arrays must be rectangular"
            )
            return [subtensor.count] + subtensorShape
        }
    }

    public subscript(_ indices: [Int]) -> S {
        switch (self, indices.first) {
        case (.scalar(let value), nil):
            return value
        case (.array(let subtensor), .some(let i)):
            precondition(i >= 0 && i < subtensor.count, "Tensor index out of bounds")
            return subtensor[i][Array(indices.dropFirst())]
        default:
            preconditionFailure("Tensor index rank must match tensor rank")
        }
    }

    public func flatten() -> [S] {
        Self.indexCombinations(for: shape).map { self[$0] }
    }

    private static func indexCombinations(for shape: [Int]) -> [[Int]] {
        // E.g., indexCombinations(for: [2, 3]) -> [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
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
}

extension TensorNestedArray: ExpressibleByArrayLiteral {
    public typealias ArrayLiteralElement = TensorNestedArray<S>
    public init(arrayLiteral elements: TensorNestedArray<S>...) {
        self = .array(elements)
    }
}

extension TensorNestedArray: ExpressibleByIntegerLiteral where S: ExpressibleByIntegerLiteral {
    public typealias IntegerLiteralType = S.IntegerLiteralType
    public init(integerLiteral value: S.IntegerLiteralType) {
        self = .scalar(S(integerLiteral: value))
    }
}

extension TensorNestedArray: ExpressibleByFloatLiteral where S: ExpressibleByFloatLiteral {
    public typealias FloatLiteralType = S.FloatLiteralType
    public init(floatLiteral value: S.FloatLiteralType) {
        self = .scalar(S(floatLiteral: value))
    }
}

// MARK: - TensorStructure

public protocol TensorStructure {
    associatedtype S: PluScalar
    var shape: [Int] { get }
    var rank: Int { get }
    init(_ values: TensorNestedArray<S>)
}
