public struct VectorDenseReference<S: PluScalar>: VectorType {
    private var values: [S]
    
    public var size: Int { values.count }
    public  subscript(i: Int) -> S {
        get { return values[i] }
        set { values[i] = newValue }
    }

    public init(_ values: [S]) {
        self.values = values
    }
    
    public func toArray(round: Bool = false) -> [S] {
        if round {
            return values.map { $0.round() }
        }
        return values
    }

    public static func + (lhs: VectorDenseReference<S>, rhs: VectorDenseReference<S>) -> VectorDenseReference<S> {
        guard lhs.size == rhs.size else {
            fatalError("Vector dimensions must match for addition")
        }
        let result = zip(lhs.values, rhs.values).map { $0 + $1 }
        return VectorDenseReference<S>(result)
    }
    
    public static prefix func - (vector: VectorDenseReference<S>) -> VectorDenseReference<S> {
        return VectorDenseReference<S>(vector.values.map { -$0 })
    }
}
