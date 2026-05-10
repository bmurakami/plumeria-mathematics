public struct TensorView<Scalar: PluScalar> {
    public var storage: TensorStorage<Scalar>
    public var offset: Int
    public var shape: [Int]
    public var strides: [Int]
    
    public init(shape: [Int]) {
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        
        self.init(shape: shape, elements: Array(repeating: .zero, count: shape.reduce(1, *)))
    }
    
    public init(shape: [Int], elements: [Scalar]) {
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        
        let count = shape.reduce(1, *)
        precondition(count == elements.count,
                     "Tensor shape \(shape) requires \(count) elements, but got \(elements.count)")
        
        let strides = Self.columnMajorStrides(for: shape)
        self.init(storage: TensorStorage(elements), offset: 0, shape: shape, strides: strides)
    }
    
    public init(storage: TensorStorage<Scalar>, offset: Int, shape: [Int], strides: [Int]) {
        precondition(offset >= 0, "Tensor view offset must be non-negative")
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        precondition(shape.count == strides.count, "Tensor shape and strides must have the same rank")
        
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.strides = strides
    }
    
    public var rank: Int { shape.count }
    public var count: Int { shape.reduce(1, *) }
    
    public subscript(_ indices: [Int]) -> Scalar {
        get { storage[linearIndex(indices)] }
        set {
            ensureUniqueStorage()
            storage[linearIndex(indices)] = newValue
        }
    }
    
    private static func columnMajorStrides(for shape: [Int]) -> [Int] {
        var strides = Array(repeating: 0, count: shape.count)
        if !shape.isEmpty {
            strides[0] = 1
            for dimension in 1..<shape.count {
                strides[dimension] = strides[dimension - 1] * shape[dimension - 1]
            }
        }
        return strides
    }
    
    private func linearIndex(_ indices: [Int]) -> Int {
        precondition(indices.count == rank, "Tensor index rank \(indices.count) does not match tensor rank \(rank)")
        
        for (dimension, index) in indices.enumerated() {
            precondition(index >= 0 && index < shape[dimension], "Tensor index out of bounds")
        }
        
        return offset + zip(indices, strides).map(*).reduce(0, +)
    }
    
    private mutating func ensureUniqueStorage() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = TensorStorage(storage.elements)
        }
    }
}
