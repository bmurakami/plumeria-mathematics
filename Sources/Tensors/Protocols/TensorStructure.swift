public enum TensorNestedArray<S: PluScalar>: Equatable {
    case scalar(S)
    indirect case array([TensorNestedArray<S>])

    public var shape: [Int] {
        switch self {
        case .scalar:
            return []
        case .array(let children):
            precondition(!children.isEmpty, "Cannot infer tensor shape from an empty nested array")
            let childShape = children[0].shape
            precondition(children.allSatisfy { $0.shape == childShape }, "Tensor nested arrays must be rectangular")
            return [children.count] + childShape
        }
    }

    public subscript(_ indices: [Int]) -> S {
        switch (self, indices.first) {
        case (.scalar(let value), nil):
            return value
        case (.array(let children), .some(let index)):
            precondition(index >= 0 && index < children.count, "Tensor index out of bounds")
            return children[index][Array(indices.dropFirst())]
        default:
            preconditionFailure("Tensor index rank must match tensor rank")
        }
    }

    public func columnMajorElements() -> [S] {
        Self.indexCombinations(for: shape).map { self[$0] }
    }

    private static func indexCombinations(for shape: [Int]) -> [[Int]] {
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

public protocol TensorStructure {
    associatedtype S: PluScalar
    var shape: [Int] { get }
    var rank: Int { get }
    init(_ values: TensorNestedArray<S>)
}
