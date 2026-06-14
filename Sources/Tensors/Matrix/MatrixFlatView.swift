public struct MatrixFlatView<Scalar: PluScalar>: Equatable {
    var view: TensorFlatView<Scalar>

    public init(view: TensorFlatView<Scalar>) {
        precondition(view.rank == 2, "MatrixFlatView requires rank 2")
        self.view = view
    }
}

// MARK: - MatrixView

extension MatrixFlatView: MatrixView {
    public var rows: Int { view.shape[0] }
    public var columns: Int { view.shape[1] }
    public var shape: [Int] { view.shape }
    public var rank: Int { view.rank }
    public var elements: [Scalar] { flattenedFromView(columnMajorOrder: true) }
    public var isContiguous: Bool { view.isContiguous }

    public init(rows: Int, columns: Int) {
        self.init(view: TensorFlatView(shape: [rows, columns]))
    }

    public init(_ values: [[Scalar]]) {
        let rows = values.count
        let columns = values[0].count
        let columnMajorOrdering: () -> [S] = { (0..<columns).flatMap { j in (0..<rows).map { i in values[i][j] }}}
        let elements = columnMajorOrdering()
        self.init(view: TensorFlatView(shape: [rows, columns], elements: elements))
    }

    public init(_ values: TensorNestedArray<Scalar>) {
        precondition(values.shape.count == 2, "Matrix nested array must have rank 2")
        self.init(view: TensorFlatView(values))
    }

    public subscript(_ indices: [Int]) -> Scalar {
        get {
            precondition(indices.count == 2, "MatrixFlatView index rank must be 2")
            return self[indices[0], indices[1]]
        }
        set {
            precondition(indices.count == 2, "MatrixFlatView index rank must be 2")
            self[indices[0], indices[1]] = newValue
        }
    }

    public subscript(i: Int, j: Int) -> Scalar {
        get { view[[i, j]] }
        set { view[[i, j]] = newValue }
    }
}

extension MatrixFlatView {
    public subscript(rows: Range<Int>, columns: Range<Int>) -> MatrixFlatView<Scalar> {
        slice(rows: SliceRange(rows), columns: SliceRange(columns))
    }

    public subscript(rows: Range<Int>, columns: TensorSliceIndex) -> MatrixFlatView<Scalar> {
        slice(rows: SliceRange(rows), columns: columns.sliceRange(dimensionSize: self.columns))
    }

    public subscript(rows: TensorSliceIndex, columns: Range<Int>) -> MatrixFlatView<Scalar> {
        slice(rows: rows.sliceRange(dimensionSize: self.rows), columns: SliceRange(columns))
    }

    public subscript(rows: TensorSliceIndex, columns: TensorSliceIndex) -> MatrixFlatView<Scalar> {
        slice(
            rows: rows.sliceRange(dimensionSize: self.rows),
            columns: columns.sliceRange(dimensionSize: self.columns)
        )
    }

    public subscript(i: Int, columns: Range<Int>) -> VectorFlatView<Scalar> {
        slice(row: i, columns: SliceRange(columns))
    }

    public subscript(i: Int, columns: TensorSliceIndex) -> VectorFlatView<Scalar> {
        slice(row: i, columns: columns.sliceRange(dimensionSize: self.columns))
    }

    public subscript(rows: Range<Int>, j: Int) -> VectorFlatView<Scalar> {
        slice(rows: SliceRange(rows), column: j)
    }

    public subscript(rows: TensorSliceIndex, j: Int) -> VectorFlatView<Scalar> {
        slice(rows: rows.sliceRange(dimensionSize: self.rows), column: j)
    }

    public func slice(rows: SliceRange, columns: SliceRange) -> MatrixFlatView<Scalar> {
        MatrixFlatView(view: view.slice(rows: rows, columns: columns))
    }

    public func slice(row: Int, columns: SliceRange) -> VectorFlatView<Scalar> {
        precondition(row >= 0 && row < rows, "Matrix row index out of bounds")
        validate(columns: columns)

        return VectorFlatView(
            view: TensorFlatView(
                storage: view.storage,
                offset: view.offset + row * view.strides[0] + columns.start * view.strides[1],
                shape: [columns.length],
                strides: [columns.step * view.strides[1]]
            )
        )
    }

    public func slice(rows: SliceRange, column: Int) -> VectorFlatView<Scalar> {
        precondition(column >= 0 && column < columns, "Matrix column index out of bounds")
        validate(rows: rows)

        return VectorFlatView(
            view: TensorFlatView(
                storage: view.storage,
                offset: view.offset + rows.start * view.strides[0] + column * view.strides[1],
                shape: [rows.length],
                strides: [rows.step * view.strides[0]]
            )
        )
    }

    public func toArray() -> [[Scalar]] {
        (0..<rows).map { i in (0..<columns).map { j in self[i, j] } }
    }
}

extension MatrixFlatView {
    public static func == (lhs: MatrixFlatView<Scalar>, rhs: MatrixFlatView<Scalar>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements
    }
}

extension MatrixFlatView {
    private func flattenedFromView(columnMajorOrder: Bool) -> [Scalar] {
        var elements = Array(repeating: Scalar.zero, count: rows * columns)
        for i in 0..<rows {
            for j in 0..<columns {
                let k = columnMajorOrder ? i + rows * j : j + columns * i
                elements[k] = self[i, j]
            }
        }
        return elements
    }

    private func validate(rows range: SliceRange) {
        let lastIndex = range.start + (range.length - 1) * range.step
        precondition(range.start <= rows, "Slice start is out of bounds")
        precondition(range.length == 0 || lastIndex < rows, "Slice end is out of bounds")
    }

    private func validate(columns range: SliceRange) {
        let lastIndex = range.start + (range.length - 1) * range.step
        precondition(range.start <= columns, "Slice start is out of bounds")
        precondition(range.length == 0 || lastIndex < columns, "Slice end is out of bounds")
    }
}
