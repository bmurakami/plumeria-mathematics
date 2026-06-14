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
    public var elements: [Scalar] { flattenedElements() }
    public var isContiguous: Bool { strides == Self.columnMajorStrides(for: shape) }
    public var contiguousElements: [Scalar]? {
        isContiguous && offset == 0 && storage.elements.count == count ? storage.elements : nil
    }

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
        get { slice(indices) }
        set { assign(newValue, to: indices) }
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

        for (dimension, i) in indices.enumerated() {
            switch i {
            case .index:
                newOffset += i.fixedIndex(dimensionSize: shape[dimension]) * strides[dimension]
            case .range, .step, .all:
                let range = i.sliceRange(dimensionSize: shape[dimension])
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

    public mutating func assign(_ replacement: TensorFlatView<Scalar>, to indices: [TensorSliceIndex]) {
        let destination = slice(indices)
        assign(replacement, to: destination)
    }

    public mutating func assign(_ replacement: TensorFlatView<Scalar>, to ranges: [SliceRange]) {
        let destination = slice(ranges)
        assign(replacement, to: destination)
    }

    mutating func assign(_ lazy: LazyMatrix<Scalar>, rows: Int, columns: Int, to ranges: [SliceRange]) {
        var destination = slice(ranges)
        let error = sliceAssignmentShapeError(destination: destination.shape, replacement: [rows, columns])
        if let error { preconditionFailure(error) }
        ensureUniqueStorage()
        destination.storage = storage
        if Scalar.self == Double.self {
            var doubleDestination = destination as! TensorFlatView<Double>
            (lazy as! LazyMatrix<Double>).assign(to: &doubleDestination)
            destination = doubleDestination as! TensorFlatView<Scalar>
        } else {
            lazy.assign(to: &destination)
        }
    }

    mutating func assign(_ lazy: LazyTensor<Scalar>, to indices: [TensorSliceIndex]) {
        var destination = slice(indices)
        let error = sliceAssignmentShapeError(destination: destination.shape, replacement: lazy.shape)
        if let error { preconditionFailure(error) }
        ensureUniqueStorage()
        destination.storage = storage
        lazy.assign(to: &destination)
    }

    func value(index0: Int, index1: Int) -> Scalar {
        storage.elements[offset + index0 * strides[0] + index1 * strides[1]]
    }

    func storageIndex(forUncheckedIndex indices: [Int]) -> Int {
        var linearIndex = offset
        for dimension in 0..<rank {
            linearIndex += indices[dimension] * strides[dimension]
        }
        return linearIndex
    }

    mutating func setValue(_ value: Scalar, index0: Int, index1: Int) {
        ensureUniqueStorage()
        storage[offset + index0 * strides[0] + index1 * strides[1]] = value
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

        for (dimension, i) in indices.enumerated() {
            precondition(i >= 0 && i < shape[dimension], "Tensor index out of bounds")
        }

        return offset + zip(indices, strides).map(*).reduce(0, +)
    }

    private func validate(range: SliceRange, dimension: Int) {
        let dimensionSize = shape[dimension]
        let lastIndex = range.start + (range.length - 1) * range.step
        precondition(range.start <= dimensionSize, "Slice start is out of bounds")
        precondition(range.length == 0 || lastIndex < dimensionSize, "Slice end is out of bounds")
    }

    private func flattenedElements() -> [Scalar] {
        var elements: [Scalar] = []
        elements.reserveCapacity(count)
        var i = Array(repeating: 0, count: rank)
        for _ in 0..<count {
            elements.append(storage.elements[storageIndex(forUncheckedIndex: i)])
            Self.increment(&i, shape: shape)
        }
        return elements
    }

    private mutating func assign(_ replacement: TensorFlatView<Scalar>, to destination: TensorFlatView<Scalar>) {
        let error = sliceAssignmentShapeError(destination: destination.shape, replacement: replacement.shape)
        if let error {
            preconditionFailure(error)
        }
        ensureUniqueStorage()
        if destination.rank == 2 {
            assignRankTwo(replacement, to: destination)
            return
        }
        var i = Array(repeating: 0, count: destination.rank)
        for _ in 0..<destination.count {
            let element = replacement.storage.elements[replacement.storageIndex(forUncheckedIndex: i)]
            storage[destination.storageIndex(forUncheckedIndex: i)] = element
            Self.increment(&i, shape: destination.shape)
        }
    }

    private mutating func assignRankTwo(_ replacement: TensorFlatView<Scalar>, to destination: TensorFlatView<Scalar>) {
        if destination.strides[0] == 1 && replacement.strides[0] == 1 && storage !== replacement.storage {
            assignRankTwoColumnSegments(replacement, to: destination)
            return
        }
        for j in 0..<destination.shape[1] {
            let destinationColumn = destination.offset + j * destination.strides[1]
            let replacementColumn = replacement.offset + j * replacement.strides[1]
            for i in 0..<destination.shape[0] {
                storage[destinationColumn + i * destination.strides[0]] =
                    replacement.storage.elements[replacementColumn + i * replacement.strides[0]]
            }
        }
    }

    private mutating func assignRankTwoColumnSegments(_ replacement: TensorFlatView<Scalar>,
                                                      to destination: TensorFlatView<Scalar>) {
        storage.elements.withUnsafeMutableBufferPointer { destinationElements in
            replacement.storage.elements.withUnsafeBufferPointer { replacementElements in
                for j in 0..<destination.shape[1] {
                    let destinationColumn = destination.offset + j * destination.strides[1]
                    let replacementColumn = replacement.offset + j * replacement.strides[1]
                    var destinationIndex = destinationColumn
                    var replacementIndex = replacementColumn
                    for _ in 0..<destination.shape[0] {
                        destinationElements[destinationIndex] = replacementElements[replacementIndex]
                        destinationIndex += 1
                        replacementIndex += 1
                    }
                }
            }
        }
    }

    private static func increment(_ i: inout [Int], shape: [Int]) {
        for dimension in 0..<shape.count {
            i[dimension] += 1
            if i[dimension] < shape[dimension] { return }
            i[dimension] = 0
        }
    }

    private mutating func ensureUniqueStorage() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = TensorStorage(storage.elements)
        }
    }
}

func sliceAssignmentShapeError(destination: [Int], replacement: [Int]) -> String? {
    if destination == replacement { return nil }
    return "Assigned slice shape \(replacement) must match destination slice shape \(destination)"
}
