public struct TensorDenseReference<S: PluScalar>: PluTensor, TensorStructure, TensorMultiplication,
    TensorArithmeticReference {
    public typealias MatrixImplementation = MatrixDenseReference<S>

    private var storage: TensorNestedArray<S>

    public let shape: [Int]
    public var rank: Int { shape.count }
    public var elements: [S] { storage.columnMajorElements() }

    public init(_ values: TensorNestedArray<S>) {
        self.shape = values.shape
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
