public struct TensorFlatView<Scalar: PluScalar>: Equatable {
    public var storage: TensorStorage<Scalar>
    public var offset: Int
    public var shape: [Int]
    public var strides: [Int]

    public init(storage: TensorStorage<Scalar>, offset: Int, shape: [Int], strides: [Int]) {
        precondition(offset >= 0, "Tensor view offset must be non-negative")
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        precondition(shape.count == strides.count, "Tensor shape and strides must have the same rank")
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.strides = strides
    }
}

// MARK: - TensorView

extension TensorFlatView: TensorView {
    public var rank: Int { shape.count }
    public var count: Int { shape.reduce(1, *) }
    public var elements: [Scalar] { (0..<count).map { self[tensorIndices(forFlatIndex: $0)] } }
    public var isContiguous: Bool { strides == Self.columnMajorStrides(for: shape) }
    public var contiguousElements: [Scalar]? { isContiguous && offset == 0 ? storage.elements : nil }

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

    public init(_ values: TensorNestedArray<Scalar>) {
        self.init(shape: values.shape, elements: values.flatten())
    }

    public subscript(_ indices: [Int]) -> Scalar {
        get { storage[linearIndex(indices)] }
        set {
            ensureUniqueStorage()
            storage[linearIndex(indices)] = newValue
        }
    }
    
    public subscript(_ indices: Int...) -> Scalar {
        get { self[indices] }
        set { self[indices] = newValue }
    }
    
    public subscript(_ indices: TensorSliceIndex...) -> TensorFlatView<Scalar> {
        slice(indices)
    }
}

extension TensorFlatView {
    public func slice(_ ranges: [SliceRange]) -> TensorFlatView<Scalar> {
        precondition(ranges.count == rank, "Slice rank must match tensor rank")
        for (dimension, range) in ranges.enumerated() {
            let dimensionSize = shape[dimension]
            let lastIndex = range.start + (range.length - 1) * range.step
            precondition(range.start <= dimensionSize, "Slice start is out of bounds")
            precondition(range.length == 0 || lastIndex < dimensionSize, "Slice end is out of bounds")
        }
        
        let newOffset = offset + zip(ranges, strides).map { $0.start * $1 }.reduce(0, +)
        let newShape = ranges.map(\.length)
        let newStrides = zip(ranges, strides).map { $0.step * $1 }
        
        return TensorFlatView(storage: storage, offset: newOffset, shape: newShape, strides: newStrides)
    }
    
    public func slice(_ range: SliceRange) -> TensorFlatView<Scalar> {
        precondition(rank == 1, "Vector slice requires rank 1")
        return slice([range])
    }
    
    public func slice(rows: SliceRange, columns: SliceRange) -> TensorFlatView<Scalar> {
        precondition(rank == 2, "Matrix slice requires rank 2")
        return slice([rows, columns])
    }
    
    public func slice(_ indices: [TensorSliceIndex]) -> TensorFlatView<Scalar> {
        precondition(indices.count == rank, "Slice rank must match tensor rank")
        
        var newOffset = offset
        var newShape: [Int] = []
        var newStrides: [Int] = []
        
        for (dimension, index) in indices.enumerated() {
            switch index {
            case .index:
                newOffset += index.fixedIndex(dimensionSize: shape[dimension]) * strides[dimension]
            case .range, .step, .all:
                let range = index.sliceRange(dimensionSize: shape[dimension])
                validate(range: range, dimension: dimension)
                newOffset += range.start * strides[dimension]
                newShape.append(range.length)
                newStrides.append(range.step * strides[dimension])
            }
        }
        
        return TensorFlatView(storage: storage, offset: newOffset, shape: newShape, strides: newStrides)
    }
    
    public func vectorSlice(_ indices: TensorSliceIndex...) -> VectorFlatView<Scalar> {
        let view = slice(indices)
        precondition(view.rank == 1, "Tensor slice result must have rank 1")
        return VectorFlatView(view: view)
    }
    
    public func matrixSlice(_ indices: TensorSliceIndex...) -> MatrixFlatView<Scalar> {
        let view = slice(indices)
        precondition(view.rank == 2, "Tensor slice result must have rank 2")
        return MatrixFlatView(view: view)
    }
}

extension TensorFlatView {
    public static func == (lhs: TensorFlatView<Scalar>, rhs: TensorFlatView<Scalar>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements
    }
}

extension TensorFlatView {
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
    
    private func validate(range: SliceRange, dimension: Int) {
        let dimensionSize = shape[dimension]
        let lastIndex = range.start + (range.length - 1) * range.step
        precondition(range.start <= dimensionSize, "Slice start is out of bounds")
        precondition(range.length == 0 || lastIndex < dimensionSize, "Slice end is out of bounds")
    }
    
    private func tensorIndices(forFlatIndex linearIndex: Int) -> [Int] {
        var remaining = linearIndex
        return shape.map { dimension in
            let index = remaining % dimension
            remaining /= dimension
            return index
        }
    }
    
    private mutating func ensureUniqueStorage() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = TensorStorage(storage.elements)
        }
    }
}
