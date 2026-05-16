public struct TensorDenseBLAS<S: PluScalar>: TensorMultiplication, TensorArithmeticBLAS, Equatable {
    public typealias MatrixImplementation = MatrixDenseBLAS<S>

    private var view: TensorFlatView<S>

    public var shape: [Int] { view.shape }
    public var rank: Int { view.rank }
    public var elements: [S] {
        get { view.contiguousElements ?? view.elements }
        set { view = TensorFlatView(shape: shape, elements: newValue) }
    }

    public init(shape: [Int], initialValue: S = .zero) {
        self.view = TensorFlatView(shape: shape, elements: Array(repeating: initialValue, count: shape.reduce(1, *)))
    }

    public init(shape: [Int], elements: [S]) {
        self.view = TensorFlatView(shape: shape, elements: elements)
    }

    public init(_ values: TensorNestedArray<S>) {
        self.init(shape: values.shape, elements: values.columnMajorElements())
    }

    public subscript(_ indices: [Int]) -> S {
        get { view[indices] }
        set { view[indices] = newValue }
    }

    public subscript(_ indices: Int...) -> S {
        get { self[indices] }
        set { self[indices] = newValue }
    }

    public static func == (left: TensorDenseBLAS<S>, right: TensorDenseBLAS<S>) -> Bool {
        left.view == right.view
    }
}
