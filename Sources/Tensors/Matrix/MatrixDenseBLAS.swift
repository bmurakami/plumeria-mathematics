import AccelerateWrapper
import OpenBLASWrapper

// Usage of MatrixArithmeticReference is temporary until determinant and inverse use LAPACK.
public struct MatrixDenseBLAS<S: PluScalar>: MatrixArithmeticReference, TensorArithmeticBLAS,
                                             MatrixColumnMajorInitializable, Equatable {
    private var view: TensorFlatView<S>
    public var blasImplementation: BLAS

    public init(rows: Int, columns: Int, values: [S]) {
        self.init(rows: rows, columns: columns, values: values, blasImplementation: .default)
    }

    init(rows: Int, columns: Int, values: [S], blasImplementation: BLAS) {
        self.view = TensorFlatView(shape: [rows, columns], elements: values)
        self.blasImplementation = blasImplementation
    }

    init(view: TensorFlatView<S>, blasImplementation: BLAS = .default) {
        precondition(view.rank == 2, "MatrixDenseBLAS view must have rank 2")
        self.view = view
        self.blasImplementation = blasImplementation
    }
}

// MARK: - PluMatrix

extension MatrixDenseBLAS: PluMatrix {
    public var rows: Int { view.shape[0] }
    public var columns: Int { view.shape[1] }
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { shape.count }
    public var elements: [S] {
        get { columnMajorStorage() }
        set {
            precondition(newValue.count == rows * columns, "Matrix element count must match matrix shape")
            view = TensorFlatView(shape: shape, elements: newValue)
        }
    }

    public subscript(i: Int, j: Int) -> S {
        get { value(row: i, column: j) }
        set { setValue(newValue, row: i, column: j) }
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

    public init(rows: Int, columns: Int, initialValue: S = S.zero) {
        let elements = Array(repeating: initialValue, count: rows * columns)
        self.view = TensorFlatView(shape: [rows, columns], elements: elements)
        self.blasImplementation = .default
    }

    public init(_ values: [[S]]) {
        let rows = values.count
        let columns = values[0].count
        let elements = (0..<columns).flatMap { column in
            (0..<rows).map { row in values[row][column] }
        }
        self.view = TensorFlatView(shape: [rows, columns], elements: elements)
        self.blasImplementation = .default
    }

    public init(_ values: TensorNestedArray<S>) {
        precondition(values.shape.count == 2, "Matrix nested array must have rank 2")
        self.view = TensorFlatView(values)
        self.blasImplementation = .default
    }

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

    public func toArray(round: Bool = false) -> [[S]] {
        (0..<rows).map { row in
            (0..<columns).map { column in
                let value = value(row: row, column: column)
                return round ? value.round() : value
            }
        }
    }

    public func flatten(columnMajorOrder: Bool = true) -> [S] {
        columnMajorOrder ? columnMajorStorage() : flattenedFromView(columnMajorOrder: false)
    }
}

extension MatrixDenseBLAS {
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
        slice(rows: rows.sliceRange(dimensionSize: self.rows), columns: columns.sliceRange(dimensionSize: self.columns))
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
}

extension MatrixDenseBLAS {
    public func times<V: PluVector>(_ v: V) -> V where V.S == S {
        precondition(columns == v.size, "Number of columns in matrix must equal size of vector")
        switch S.self {
        case is Double.Type:
            let A = columnMajorStorage() as! [Double]
            let x = vectorElements(v) as! [Double]
            var y = Array(repeating: 0.0, count: rows)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.dgemv(Int32(rows), Int32(columns), A, x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.dgemv(Int32(rows), Int32(columns), A, x, &y)
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
            let A = columnMajorStorage() as! [Double]
            let B = columnMajorElements(from: m) as! [Double]
            var C = Array(repeating: 0.0, count: rows * m.columns)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.dgemm(Int32(rows), Int32(m.columns), Int32(columns), A, B, &C)
            #endif
            case .openBLAS:
                OpenBLASOperations.dgemm(Int32(rows), Int32(m.columns), Int32(columns), A, B, &C)
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
            fatalError("Unsupported scalar type")
        }
    }
}

extension MatrixDenseBLAS {
    public func transpose() -> MatrixDenseBLAS<S> {
        var transposed = MatrixDenseBLAS(rows: columns, columns: rows)
        for row in 0..<rows {
            for column in 0..<columns {
                transposed.setValue(value(row: row, column: column), row: column, column: row)
            }
        }
        return transposed
    }
}

extension MatrixDenseBLAS {
    public static func == (lhs: MatrixDenseBLAS<S>, rhs: MatrixDenseBLAS<S>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements && lhs.blasImplementation == rhs.blasImplementation
    }
}

extension MatrixDenseBLAS {
    private func value(row: Int, column: Int) -> S { view[[row, column]] }
    private mutating func setValue(_ value: S, row: Int, column: Int) { view[[row, column]] = value }

    private func flattenedFromView(columnMajorOrder: Bool) -> [S] {
        var elements = Array(repeating: S.zero, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                let index = columnMajorOrder ? row + rows * column : column + columns * row
                elements[index] = value(row: row, column: column)
            }
        }
        return elements
    }

    private func columnMajorStorage() -> [S] {
        view.contiguousElements ?? flattenedFromView(columnMajorOrder: true)
    }

    private func vectorElements<V: PluVector>(_ vector: V) -> [S] where V.S == S {
        if let vector = vector as? VectorDenseBLAS<S> { return vector.elements }
        return vector.toArray(round: false)
    }

    private func columnMajorElements<M: PluMatrix>(from matrix: M) -> [S] where M.S == S {
        if let matrix = matrix as? MatrixDenseBLAS<S> { return matrix.columnMajorStorage() }
        return matrix.flatten(columnMajorOrder: true)
    }

    private static func interleaved(_ values: [Complex]) -> [Double] {
        values.flatMap { [$0.real, $0.imaginary] }
    }

    private static func complexValues(_ values: [Double]) -> [Complex] {
        stride(from: 0, to: values.count, by: 2).map { Complex(values[$0], values[$0 + 1]) }
    }

}

extension MatrixDenseBLAS: MatrixEigen where S == Double {
    public typealias Eigenvectors = MatrixDenseBLAS<Complex>

    public func eigen() -> Eigen<MatrixDenseBLAS<Complex>> {
        precondition(rows == columns, "Eigen decomposition requires a square matrix")
        let n = Int32(rows)
        var matrix = flatten()
        var real = Array(repeating: 0.0, count: rows)
        var imaginary = Array(repeating: 0.0, count: rows)
        var vectors = Array(repeating: 0.0, count: rows * columns)
        #if canImport(Accelerate)
        let info = AccelerateOperations.dgeev(n, &matrix, &real, &imaginary, &vectors)
        #else
        let info = OpenBLASOperations.dgeev(n, &matrix, &real, &imaginary, &vectors)
        #endif
        precondition(info == 0, "Eigen decomposition failed with LAPACK info \(info)")
        return Eigen(values: eigenvalues(real: real, imaginary: imaginary),
                     vectors: eigenvectors(real: real, imaginary: imaginary, vectors: vectors))
    }

    private func eigenvalues(real: [Double], imaginary: [Double]) -> [Complex] {
        zip(real, imaginary).map { Complex($0, $1) }
    }

    private func eigenvectors(real: [Double], imaginary: [Double], vectors: [Double]) -> MatrixDenseBLAS<Complex> {
        var result = MatrixDenseBLAS<Complex>(rows: rows, columns: columns, initialValue: .zero)
        var column = 0
        while column < columns {
            if imaginary[column] == 0.0 {
                for row in 0..<rows {
                    result[row, column] = Complex(vectors[row + rows * column], 0.0)
                }
                column += 1
            } else {
                for row in 0..<rows {
                    let realPart = vectors[row + rows * column]
                    let imaginaryPart = vectors[row + rows * (column + 1)]
                    result[row, column] = Complex(realPart, imaginaryPart)
                    result[row, column + 1] = Complex(realPart, -imaginaryPart)
                }
                column += 2
            }
        }
        return result
    }
}
