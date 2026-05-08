public struct VectorDenseReference<S: PluScalar>: PluVector, FlatTensor {
    public typealias Scalar = S
    
    public var elements: [S]
    
    // MARK: - PluVector conformance
    public var size: Int { elements.count }
    public  subscript(i: Int) -> S {
        get { return elements[i] }
        set { elements[i] = newValue }
    }

    public init(_ values: [S]) {
        self.elements = values
    }
    
    public func toArray(round: Bool) -> [S] {
        if round {
            return elements.map { $0.round() }
        }
        return elements
    }
    
    // MARK: - FlatTensor conformance
    public var shape: [Int] { [size] }
    
    public init(shape: [Int]) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] >= 0, "Vector size must be non-negative")
        
        self.init(Array(repeating: .zero, count: shape[0]))
    }
    
    public init(shape: [Int], elements: [S]) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] == elements.count, "Vector shape \(shape) requires \(shape[0]) elements, but got \(elements.count)")
        
        self.init(elements)
    }

    // MARK: - PluTensor conformance
    public static func + (lhs: VectorDenseReference<S>, rhs: VectorDenseReference<S>) -> VectorDenseReference<S> {
        guard lhs.size == rhs.size else {
            fatalError("Vector dimensions must match for addition")
        }
        let result = zip(lhs.elements, rhs.elements).map { $0 + $1 }
        return VectorDenseReference<S>(result)
    }
    
    public static prefix func - (vector: VectorDenseReference<S>) -> VectorDenseReference<S> {
        return VectorDenseReference<S>(vector.elements.map { -$0 })
    }
}
