public indirect enum TensorNestedArray<S: PluScalar>: Equatable {
    case scalar(S)
    case array([TensorNestedArray<S>])
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

public struct TensorDenseReference<S: PluScalar>: PluTensor, TensorStructure, TensorArithmeticReference {
    private var storage: TensorNestedArray<S>

    public let shape: [Int]
    public var rank: Int { shape.count }

    public init(_ values: TensorNestedArray<S>) {
        self.shape = Self.shape(of: values)
        self.storage = values
    }

    public init(shape: [Int], initialValue: S = .zero) {
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        self.shape = shape
        self.storage = Self.storage(shape: shape, initialValue: initialValue)
    }

    public subscript(_ indices: [Int]) -> S {
        get {
            precondition(indices.count == rank, "Tensor index rank must match tensor rank")
            return Self.value(in: storage, at: indices)
        }
        set {
            precondition(indices.count == rank, "Tensor index rank must match tensor rank")
            Self.setValue(newValue, in: &storage, at: indices)
        }
    }

    public subscript(_ indices: Int...) -> S {
        get { self[indices] }
        set { self[indices] = newValue }
    }

    public func toNestedArray() -> TensorNestedArray<S> { storage }

    private static func shape(of storage: TensorNestedArray<S>) -> [Int] {
        switch storage {
        case .scalar:
            return []
        case .array(let children):
            precondition(!children.isEmpty, "Cannot infer tensor shape from an empty nested array")
            let childShape = shape(of: children[0])
            precondition(
                children.allSatisfy { shape(of: $0) == childShape },
                "Tensor nested arrays must be rectangular"
            )
            return [children.count] + childShape
        }
    }

    private static func storage(shape: [Int], initialValue: S) -> TensorNestedArray<S> {
        guard let size = shape.first else { return .scalar(initialValue) }
        return .array((0..<size).map { _ in storage(shape: Array(shape.dropFirst()), initialValue: initialValue) })
    }

    private static func value(in storage: TensorNestedArray<S>, at indices: [Int]) -> S {
        guard let index = indices.first else {
            guard case .scalar(let value) = storage else {
                preconditionFailure("Tensor index rank must match tensor rank")
            }
            return value
        }
        guard case .array(let children) = storage else {
            preconditionFailure("Tensor index rank must match tensor rank")
        }
        precondition(index >= 0 && index < children.count, "Tensor index out of bounds")
        return value(in: children[index], at: Array(indices.dropFirst()))
    }

    private static func setValue(_ value: S, in storage: inout TensorNestedArray<S>, at indices: [Int]) {
        guard let index = indices.first else {
            storage = .scalar(value)
            return
        }
        guard case .array(var children) = storage else {
            preconditionFailure("Tensor index rank must match tensor rank")
        }
        precondition(index >= 0 && index < children.count, "Tensor index out of bounds")
        setValue(value, in: &children[index], at: Array(indices.dropFirst()))
        storage = .array(children)
    }
}
