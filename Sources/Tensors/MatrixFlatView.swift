public struct MatrixFlatView<Scalar: PluScalar>: MatrixView, Equatable {
    var view: TensorFlatView<Scalar>
    public var rows: Int { view.shape[0] }
    public var columns: Int { view.shape[1] }
    public var shape: [Int] { view.shape }
    public var rank: Int { view.rank }
    public var elements: [Scalar] { viewElements(columnMajorOrder: true) }
    public var isContiguous: Bool { view.isContiguous }
    
    public init(view: TensorFlatView<Scalar>) {
        precondition(view.rank == 2, "MatrixFlatView requires rank 2")
        self.view = view
    }
    
    public init(rows: Int, columns: Int) {
        self.init(view: TensorFlatView(shape: [rows, columns]))
    }
    
    public init(_ values: [[Scalar]]) {
        let rows = values.count
        let columns = values[0].count
        let elements = (0..<columns).flatMap { column in (0..<rows).map { row in values[row][column] } }
        self.init(view: TensorFlatView(shape: [rows, columns], elements: elements))
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
    
    public subscript(row: Int, column: Int) -> Scalar {
        get { view[[row, column]] }
        set { view[[row, column]] = newValue }
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
        (0..<rows).map { row in (0..<columns).map { column in self[row, column] } }
    }
    
    public static func == (lhs: MatrixFlatView<Scalar>, rhs: MatrixFlatView<Scalar>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements
    }
    
    private func viewElements(columnMajorOrder: Bool) -> [Scalar] {
        var elements = Array(repeating: Scalar.zero, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                let index = columnMajorOrder ? row + rows * column : column + columns * row
                elements[index] = self[row, column]
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
