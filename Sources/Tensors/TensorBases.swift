public struct VectorBase<Implementation: PluVector>: PluVector, TensorArithmeticReference {
    public typealias S = Implementation.S

    private var implementation: Implementation
    public var size: Int { implementation.size }

    public subscript(index: Int) -> Implementation.S {
        get { implementation[index] }
        set { implementation[index] = newValue }
    }

    public subscript(_ indices: [Int]) -> Implementation.S {
        get {
            precondition(indices.count == 1, "Vector index rank must be 1")
            return implementation[indices[0]]
        }
        set {
            precondition(indices.count == 1, "Vector index rank must be 1")
            implementation[indices[0]] = newValue
        }
    }

    public init(_ elements: [Implementation.S]) {
        self.implementation = Implementation(elements)
    }

    public init(shape: [Int], initialValue: Implementation.S) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] >= 0, "Vector size must be non-negative")

        self.implementation = Implementation(Array(repeating: initialValue, count: shape[0]))
    }

    public init(_ implementation: Implementation) {
        self.implementation = implementation
    }

    public func toArray(round: Bool) -> [Implementation.S] { implementation.toArray(round: round) }
}

public struct MatrixBase<Implementation: PluMatrix>: PluMatrix, TensorArithmeticReference {
    public typealias S = Implementation.S

    private var implementation: Implementation
    public var rows: Int { implementation.rows }
    public var columns: Int { implementation.columns }

    public subscript(i: Int, j: Int) -> Implementation.S {
        get { implementation[i, j] }
        set { implementation[i, j] = newValue }
    }

    public subscript(_ indices: [Int]) -> Implementation.S {
        get {
            precondition(indices.count == 2, "Matrix index rank must be 2")
            return implementation[indices[0], indices[1]]
        }
        set {
            precondition(indices.count == 2, "Matrix index rank must be 2")
            implementation[indices[0], indices[1]] = newValue
        }
    }

    public init(rows: Int, columns: Int, initialValue: Implementation.S = .zero) {
        self.implementation = Implementation(rows: rows, columns: columns, initialValue: initialValue)
    }

    public init(shape: [Int], initialValue: Implementation.S) {
        precondition(shape.count == 2, "Matrix shape must have rank 2")
        precondition(shape.allSatisfy { $0 >= 0 }, "Matrix shape dimensions must be non-negative")

        self.init(rows: shape[0], columns: shape[1], initialValue: initialValue)
    }

    public init(_ values: [[Implementation.S]]) {
        self.implementation = Implementation(values)
    }

    public init(_ implementation: Implementation) {
        self.implementation = implementation
    }

    public func times<V: PluVector>(_ v: V) -> V where V.S == Implementation.S { implementation.times(v) }
    public func times<M: PluMatrix>(_ m: M) -> MatrixBase<Implementation> where M.S == Implementation.S {
        MatrixBase(implementation.times(m))
    }

    public func transpose() -> MatrixBase<Implementation> {
        MatrixBase(implementation.transpose())
    }

    public func toArray(round: Bool) -> [[Implementation.S]] { implementation.toArray(round: round) }
    public func flatten(columnMajorOrder: Bool) -> [Implementation.S] {
        implementation.flatten(columnMajorOrder: columnMajorOrder)
    }
}
