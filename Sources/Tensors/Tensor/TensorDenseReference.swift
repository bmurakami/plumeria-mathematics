public struct TensorDenseReference<S: PluScalar>: TensorArithmeticReference, PluTensor {
    private var storage: TensorNestedArray<S>

    public let shape: [Int]
    public var rank: Int { shape.count }
    public var elements: [S] { storage.flatten() }
}

// MARK: - TensorStructure

extension TensorDenseReference: TensorStructure {
    public init(_ elements: TensorNestedArray<S>) {
        self.shape = elements.shape
        self.storage = elements
    }

    public init(shape: [Int], initialValue: S = .zero) {
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        self.shape = shape
        self.storage = Self.storage(shape: shape, initialValue: initialValue)
    }

    public init(shape: [Int], elements: [S]) {
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        let count = shape.reduce(1, *)
        precondition(count == elements.count,
                     "Tensor shape \(shape) requires \(count) elements, but got \(elements.count)")
        self.init(shape: shape, initialValue: .zero)
        for (index, element) in zip(Self.indexCombinations(for: shape), elements) {
            self[index] = element
        }
    }
}

extension TensorDenseReference {
    public func flatten() -> [S] { elements }

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
}

// MARK: - TensorMultiplication

extension TensorDenseReference: TensorMultiplication {
    public typealias MatrixImplementation = MatrixDenseReference<S>
}

extension TensorDenseReference {
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
        guard case .array(let subtensors) = storage else {
            preconditionFailure("Tensor index rank must match tensor rank")
        }
        precondition(index >= 0 && index < subtensors.count, "Tensor index out of bounds")
        return value(in: subtensors[index], at: Array(indices.dropFirst()))
    }

    private static func setValue(_ value: S, in storage: inout TensorNestedArray<S>, at indices: [Int]) {
        guard let index = indices.first else {
            storage = .scalar(value)
            return
        }
        guard case .array(var subtensors) = storage else {
            preconditionFailure("Tensor index rank must match tensor rank")
        }
        precondition(index >= 0 && index < subtensors.count, "Tensor index out of bounds")
        setValue(value, in: &subtensors[index], at: Array(indices.dropFirst()))
        storage = .array(subtensors)
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
