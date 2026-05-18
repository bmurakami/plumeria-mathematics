public struct VectorDenseReference<S: PluScalar>: TensorArithmeticReference {
    public var elements: [S]

    public init(_ values: [S]) {
        self.elements = values
    }
}

// MARK: - PluVector

extension VectorDenseReference: PluVector {
    public var size: Int { elements.count }

    public subscript(i: Int) -> S {
        get { return elements[i] }
        set { elements[i] = newValue }
    }

    public subscript(_ indices: [Int]) -> S {
        get {
            precondition(indices.count == 1, "Vector index rank must be 1")
            return self[indices[0]]
        }
        set {
            precondition(indices.count == 1, "Vector index rank must be 1")
            self[indices[0]] = newValue
        }
    }

    public init(_ values: TensorNestedArray<S>) {
        precondition(values.shape.count == 1, "Vector nested array must have rank 1")
        self.init(values.flatten())
    }

    public func toArray(round: Bool) -> [S] {
        if round {
            return elements.map { $0.round() }
        }
        return elements
    }

    public var shape: [Int] { [size] }
    public var rank: Int { shape.count }

    public init(shape: [Int]) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] >= 0, "Vector size must be non-negative")

        self.init(Array(repeating: .zero, count: shape[0]))
    }

    public init(shape: [Int], initialValue: S) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] >= 0, "Vector size must be non-negative")

        self.init(Array(repeating: initialValue, count: shape[0]))
    }

    public init(shape: [Int], elements: [S]) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] == elements.count,
                     "Vector shape \(shape) requires \(shape[0]) elements, but got \(elements.count)")

        self.init(elements)
    }
}
