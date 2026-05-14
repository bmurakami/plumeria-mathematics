import AccelerateWrapper
import OpenBLASWrapper

public struct MatrixDenseBLAS<S: PluScalar>: PluMatrix, TensorElementwiseArithmetic {
    private var view: TensorFlatView<S>
    public var blasImplementation: BLAS
    
    private func value(row: Int, column: Int) -> S { view[[row, column]] }
    private mutating func setValue(_ value: S, row: Int, column: Int) { view[[row, column]] = value }

    init(rows: Int, columns: Int, values: [S], blasImplementation: BLAS = BLAS.default) {
        self.view = TensorFlatView(shape: [rows, columns], elements: values)
        self.blasImplementation = blasImplementation
    }
    
    init(view: TensorFlatView<S>, blasImplementation: BLAS = BLAS.default) {
        precondition(view.rank == 2, "MatrixDenseBLAS view must have rank 2")
        self.view = view
        self.blasImplementation = blasImplementation
    }
    
    // MARK: - PluMatrix conformance
    public var rows: Int { view.shape[0] }
    public var columns: Int { view.shape[1] }
    
    public subscript(i: Int, j: Int) -> S {
        get { value(row: i, column: j) }
        set { setValue(newValue, row: i, column: j) }
    }
    
    public init(rows: Int, columns: Int, initialValue: S = S.zero) {
        let elements = Array(repeating: initialValue, count: rows * columns)
        self.view = TensorFlatView(shape: [rows, columns], elements: elements)
        self.blasImplementation = BLAS.default
    }

    public init(_ values: [[S]]) {
        let rows = values.count
        let columns = values[0].count
        let elements = (0..<columns).flatMap { column in
            (0..<rows).map { row in
                values[row][column]
            }
        }
        self.view = TensorFlatView(shape: [rows, columns], elements: elements)
        self.blasImplementation = .openBLAS
    }

    public var elements: [S] {
        get { viewElements(columnMajorOrder: true) }
        set {
            precondition(newValue.count == rows * columns, "Matrix element count must match matrix shape")
            view = TensorFlatView(shape: shape, elements: newValue)
        }
    }
    
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { shape.count }
    
    public init(shape: [Int]) {
        precondition(shape.count == 2, "MatrixDenseBLAS shape must have rank 2")
        precondition(shape.allSatisfy { $0 >= 0 }, "Matrix shape dimensions must be non-negative")
        
        self.init(rows: shape[0], columns: shape[1])
    }

    public init(shape: [Int], initialValue: S) {
        precondition(shape.count == 2, "MatrixDenseBLAS shape must have rank 2")
        precondition(shape.allSatisfy { $0 >= 0 }, "Matrix shape dimensions must be non-negative")

        self.init(rows: shape[0], columns: shape[1], initialValue: initialValue)
    }
    
    public init(shape: [Int], elements: [S]) {
        precondition(shape.count == 2, "MatrixDenseBLAS shape must have rank 2")
        precondition(shape.allSatisfy { $0 >= 0 }, "Matrix shape dimensions must be non-negative")
        let count = shape.reduce(1, *)
        precondition(count == elements.count,
                     "Matrix shape \(shape) requires \(count) elements, but got \(elements.count)")
        
        self.init(rows: shape[0], columns: shape[1], values: elements)
    }
    
    public subscript(_ indices: [Int]) -> S {
        get {
            precondition(indices.count == 2, "MatrixDenseBLAS index rank must be 2")
            precondition(indices[0] >= 0 && indices[0] < rows, "Matrix row index out of bounds")
            precondition(indices[1] >= 0 && indices[1] < columns, "Matrix column index out of bounds")
            
            return value(row: indices[0], column: indices[1])
        }
        set {
            precondition(indices.count == 2, "MatrixDenseBLAS index rank must be 2")
            precondition(indices[0] >= 0 && indices[0] < rows, "Matrix row index out of bounds")
            precondition(indices[1] >= 0 && indices[1] < columns, "Matrix column index out of bounds")
            
            setValue(newValue, row: indices[0], column: indices[1])
        }
    }
    
    public func slice(rows: SliceRange, columns: SliceRange) -> MatrixDenseBLAS<S> {
        MatrixDenseBLAS(view: view.slice(rows: rows, columns: columns), blasImplementation: blasImplementation)
    }
    
    public func slice(row: Int, columns: SliceRange) -> VectorFlatView<S> {
        MatrixFlatView(view: view).slice(row: row, columns: columns)
    }
    
    public func slice(rows: SliceRange, column: Int) -> VectorFlatView<S> {
        MatrixFlatView(view: view).slice(rows: rows, column: column)
    }
    
    public subscript(rows: Range<Int>, columns: Range<Int>) -> MatrixDenseBLAS<S> {
        slice(rows: SliceRange(rows), columns: SliceRange(columns))
    }
    
    public subscript(rows: Range<Int>, columns: TensorSliceIndex) -> MatrixDenseBLAS<S> {
        slice(rows: SliceRange(rows), columns: columns.sliceRange(dimensionSize: self.columns))
    }
    
    public subscript(rows: TensorSliceIndex, columns: Range<Int>) -> MatrixDenseBLAS<S> {
        slice(rows: rows.sliceRange(dimensionSize: self.rows), columns: SliceRange(columns))
    }
    
    public subscript(rows: TensorSliceIndex, columns: TensorSliceIndex) -> MatrixDenseBLAS<S> {
        slice(
            rows: rows.sliceRange(dimensionSize: self.rows),
            columns: columns.sliceRange(dimensionSize: self.columns)
        )
    }
    
    public subscript(row: Int, columns: Range<Int>) -> VectorFlatView<S> {
        slice(row: row, columns: SliceRange(columns))
    }
    
    public subscript(row: Int, columns: TensorSliceIndex) -> VectorFlatView<S> {
        slice(row: row, columns: columns.sliceRange(dimensionSize: self.columns))
    }
    
    public subscript(rows: Range<Int>, column: Int) -> VectorFlatView<S> {
        slice(rows: SliceRange(rows), column: column)
    }
    
    public subscript(rows: TensorSliceIndex, column: Int) -> VectorFlatView<S> {
        slice(rows: rows.sliceRange(dimensionSize: self.rows), column: column)
    }

    public func times<V: PluVector>(_ v: V) -> V where V.S == S {
        precondition(columns == v.size, "Number of columns in matrix must equal size of vector")
        
        switch S.self {
        case is Double.Type:
            var A = flatten() as! [Double] // Ax = y
            var x = v.toArray(round: false) as! [Double]
            var y = Array(repeating: 0.0, count: rows)

            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.dgemv(Int32(rows), Int32(columns), &A, &x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.dgemv(Int32(rows), Int32(columns), &A, &x, &y)
            }

            return V(y as! [S])
        case is Complex.Type:
            var A = MatrixDenseBLAS.interleaved(flatten() as! [Complex])
            var x = MatrixDenseBLAS.interleaved(v.toArray(round: false) as! [Complex])
            var y = Array(repeating: 0.0, count: rows * 2)

            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.zgemv(Int32(rows), Int32(columns), &A, &x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.zgemv(Int32(rows), Int32(columns), &A, &x, &y)
            }

            return V(MatrixDenseBLAS.complexValues(y) as! [S])
        default:
            fatalError("Unsupported scalar type")
        }
    }

    public func times<M: PluMatrix>(_ m: M) -> MatrixDenseBLAS<S> where M.S == S {
        precondition(columns == m.rows, "Number of matrix columns must match matrix rows")
        switch S.self {
        case is Double.Type:
            var A = flatten() as! [Double]
            var B = m.flatten() as! [Double]
            var C = Array(repeating: 0.0, count: rows * m.columns)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.dgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            #endif
            case .openBLAS:
                OpenBLASOperations.dgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            }
            return MatrixDenseBLAS(rows: rows, columns: m.columns, values: C as! [S])
        case is Complex.Type:
            var A = MatrixDenseBLAS.interleaved(flatten() as! [Complex])
            var B = MatrixDenseBLAS.interleaved(m.flatten() as! [Complex])
            var C = Array(repeating: 0.0, count: rows * m.columns * 2)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.zgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            #endif
            case .openBLAS:
                OpenBLASOperations.zgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            }
            return MatrixDenseBLAS(rows: rows, columns: m.columns, values: MatrixDenseBLAS.complexValues(C) as! [S])
        default:
            fatalError("Not yet implemented")
        }
    }

    public func transpose() -> MatrixDenseBLAS<S> {
        var transposed = MatrixDenseBLAS(rows: columns, columns: rows)
        for row in 0..<rows {
            for column in 0..<columns {
                transposed.setValue(value(row: row, column: column), row: column, column: row)
            }
        }
        return transposed
    }
 
    public func toArray(round: Bool = false) -> [[S]] {
        (0..<rows).map { row in
            (0..<columns).map { column in
                let value = value(row: row, column: column)
                return round ? value.round() : value
            }
        }
    }
    
    public func flatten(columnMajorOrder: Bool = true) -> [S] {
        viewElements(columnMajorOrder: columnMajorOrder)
    }
    
    public static func == (lhs: MatrixDenseBLAS<S>, rhs: MatrixDenseBLAS<S>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements && lhs.blasImplementation == rhs.blasImplementation
    }
    
    private func viewElements(columnMajorOrder: Bool) -> [S] {
        var elements = Array(repeating: S.zero, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                let index = columnMajorOrder ? row + rows * column : column + columns * row
                elements[index] = value(row: row, column: column)
            }
        }
        return elements
    }

    private static func interleaved(_ values: [Complex]) -> [Double] {
        values.flatMap { [$0.real, $0.imaginary] }
    }

    private static func complexValues(_ values: [Double]) -> [Complex] {
        stride(from: 0, to: values.count, by: 2).map { Complex(values[$0], values[$0 + 1]) }
    }
}
