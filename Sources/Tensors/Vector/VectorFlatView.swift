public struct VectorFlatView<Scalar: PluScalar>: Equatable {
    var view: TensorFlatView<Scalar>

    public init(view: TensorFlatView<Scalar>) {
        precondition(view.rank == 1, "VectorFlatView requires rank 1")
        self.view = view
    }
}

// MARK: - VectorView

extension VectorFlatView: VectorView {
    public var size: Int { view.shape[0] }
    public var shape: [Int] { view.shape }
    public var rank: Int { view.rank }
    public var elements: [Scalar] { (0..<size).map { self[$0] } }
    public var isContiguous: Bool { view.isContiguous }

    public init(size: Int) {
        self.init(view: TensorFlatView(shape: [size]))
    }

    public init(_ elements: [Scalar]) {
        self.init(view: TensorFlatView(shape: [elements.count], elements: elements))
    }

    public init(_ values: TensorNestedArray<Scalar>) {
        precondition(values.shape.count == 1, "Vector nested array must have rank 1")
        self.init(values.flatten())
    }

    public subscript(_ indices: [Int]) -> Scalar {
        get {
            precondition(indices.count == 1, "VectorFlatView index rank must be 1")
            return self[indices[0]]
        }
        set {
            precondition(indices.count == 1, "VectorFlatView index rank must be 1")
            self[indices[0]] = newValue
        }
    }

    public subscript(i: Int) -> Scalar {
        get { view[[i]] }
        set { view[[i]] = newValue }
    }

    public subscript(range: Range<Int>) -> VectorFlatView<Scalar> {
        slice(SliceRange(range))
    }

    public subscript(i: TensorSliceIndex) -> VectorFlatView<Scalar> {
        slice(i.sliceRange(dimensionSize: size))
    }

    public func slice(_ range: SliceRange) -> VectorFlatView<Scalar> {
        VectorFlatView(view: view.slice(range))
    }
}

extension VectorFlatView {
    public static func == (lhs: VectorFlatView<Scalar>, rhs: VectorFlatView<Scalar>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements
    }
}
