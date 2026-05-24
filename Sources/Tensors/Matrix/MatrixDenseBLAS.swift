import AccelerateWrapper
import Numerics
import OpenBLASWrapper

public struct MatrixDenseBLAS<S: PluScalar>: TensorArithmeticBLAS, MatrixColumnMajorInitializable, Equatable {
    var view: TensorFlatView<S>
    var lazy: LazyMatrix<S>?
    public var blasImplementation: BLAS

    public init(rows: Int, columns: Int, values: [S]) {
        self.init(rows: rows, columns: columns, values: values, blasImplementation: .default)
    }

    init(rows: Int, columns: Int, values: [S], blasImplementation: BLAS) {
        self.view = TensorFlatView(shape: [rows, columns], elements: values)
        self.lazy = nil
        self.blasImplementation = blasImplementation
    }

    init(view: TensorFlatView<S>, blasImplementation: BLAS = .default) {
        precondition(view.rank == 2, "MatrixDenseBLAS view must have rank 2")
        self.view = view
        self.lazy = nil
        self.blasImplementation = blasImplementation
    }

    init(rows: Int, columns: Int, lazy: LazyMatrix<S>, blasImplementation: BLAS = .default) {
        self.view = TensorFlatView(storage: TensorStorage([]), offset: 0, shape: [rows, columns], strides: [1, rows])
        self.lazy = lazy
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
            lazy = nil
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
        self.lazy = nil
        self.blasImplementation = .default
    }

    public init(_ values: [[S]]) {
        let rows = values.count
        let columns = values[0].count
        let elements = (0..<columns).flatMap { column in
            (0..<rows).map { row in values[row][column] }
        }
        self.view = TensorFlatView(shape: [rows, columns], elements: elements)
        self.lazy = nil
        self.blasImplementation = .default
    }

    public init(_ values: TensorNestedArray<S>) {
        precondition(values.shape.count == 2, "Matrix nested array must have rank 2")
        self.view = TensorFlatView(values)
        self.lazy = nil
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
        if let lazy {
            let elements = lazy.materializedElements(rows: rows, columns: columns)
            if columnMajorOrder { return elements }
            return rowMajorElements(fromColumnMajorElements: elements)
        }
        return columnMajorOrder ? columnMajorStorage() : flattenedFromView(columnMajorOrder: false)
    }
}

extension MatrixDenseBLAS {
    public func slice(rows: SliceRange, columns: SliceRange) -> MatrixDenseBLAS<S> {
        let source = materializedView()
        return MatrixDenseBLAS(view: source.slice(rows: rows, columns: columns), blasImplementation: blasImplementation)
    }

    public func slice(row: Int, columns: SliceRange) -> VectorFlatView<S> {
        MatrixFlatView(view: view).slice(row: row, columns: columns)
    }

    public func slice(rows: SliceRange, column: Int) -> VectorFlatView<S> {
        MatrixFlatView(view: view).slice(rows: rows, column: column)
    }

    public subscript(rows: Range<Int>, columns: Range<Int>) -> MatrixDenseBLAS<S> {
        get { slice(rows: SliceRange(rows), columns: SliceRange(columns)) }
        set { assign(newValue, to: [SliceRange(rows), SliceRange(columns)]) }
    }

    public subscript(rows: Range<Int>, columns: TensorSliceIndex) -> MatrixDenseBLAS<S> {
        get { slice(rows: SliceRange(rows), columns: columns.sliceRange(dimensionSize: self.columns)) }
        set { assign(newValue, to: [SliceRange(rows), columns.sliceRange(dimensionSize: self.columns)]) }
    }

    public subscript(rows: TensorSliceIndex, columns: Range<Int>) -> MatrixDenseBLAS<S> {
        get { slice(rows: rows.sliceRange(dimensionSize: self.rows), columns: SliceRange(columns)) }
        set { assign(newValue, to: [rows.sliceRange(dimensionSize: self.rows), SliceRange(columns)]) }
    }

    public subscript(rows: TensorSliceIndex, columns: TensorSliceIndex) -> MatrixDenseBLAS<S> {
        get {
            let rowRange = rows.sliceRange(dimensionSize: self.rows)
            let columnRange = columns.sliceRange(dimensionSize: self.columns)
            return slice(rows: rowRange, columns: columnRange)
        }
        set {
            let rowRange = rows.sliceRange(dimensionSize: self.rows)
            let columnRange = columns.sliceRange(dimensionSize: self.columns)
            assign(newValue, to: [rowRange, columnRange])
        }
    }

    public subscript(row: Int, columns: Range<Int>) -> VectorFlatView<S> {
        get { slice(row: row, columns: SliceRange(columns)) }
        set { view.assign(newValue.view, to: [.index(row), TensorSliceIndex.range(columns)]) }
    }

    public subscript(row: Int, columns: TensorSliceIndex) -> VectorFlatView<S> {
        get { slice(row: row, columns: columns.sliceRange(dimensionSize: self.columns)) }
        set { view.assign(newValue.view, to: [.index(row), columns]) }
    }

    public subscript(rows: Range<Int>, column: Int) -> VectorFlatView<S> {
        get { slice(rows: SliceRange(rows), column: column) }
        set { view.assign(newValue.view, to: [TensorSliceIndex.range(rows), .index(column)]) }
    }

    public subscript(rows: TensorSliceIndex, column: Int) -> VectorFlatView<S> {
        get { slice(rows: rows.sliceRange(dimensionSize: self.rows), column: column) }
        set { view.assign(newValue.view, to: [rows, .index(column)]) }
    }
}

extension MatrixDenseBLAS {
    public static func + (lhs: MatrixDenseBLAS<S>, rhs: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return doubleMatrixSum(lhs as! MatrixDenseBLAS<Double>, rhs as! MatrixDenseBLAS<Double>)
                as! MatrixDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatMatrixSum(lhs as! MatrixDenseBLAS<Float>, rhs as! MatrixDenseBLAS<Float>)
                as! MatrixDenseBLAS<S>
        }
        if lhs.isWholeMaterializedMatrix && rhs.isWholeMaterializedMatrix {
            return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                                   values: lhs.sum(lhs.view.storage.elements, rhs.view.storage.elements),
                                   blasImplementation: lhs.blasImplementation)
        }
        return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                               lazy: lhs.lazyMatrix.adding(rhs.lazyMatrix),
                               blasImplementation: lhs.blasImplementation)
    }

    public static func - (lhs: MatrixDenseBLAS<S>, rhs: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return doubleMatrixDifference(lhs as! MatrixDenseBLAS<Double>, rhs as! MatrixDenseBLAS<Double>)
                as! MatrixDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatMatrixDifference(lhs as! MatrixDenseBLAS<Float>, rhs as! MatrixDenseBLAS<Float>)
                as! MatrixDenseBLAS<S>
        }
        return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                               lazy: lhs.lazyMatrix.subtracting(rhs.lazyMatrix),
                               blasImplementation: lhs.blasImplementation)
    }

    public static prefix func - (operand: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        operand * -1
    }

    public static func * (matrix: MatrixDenseBLAS<S>, scalar: S) -> MatrixDenseBLAS<S> {
        if S.self == Double.self {
            return doubleMatrixScale(matrix as! MatrixDenseBLAS<Double>, by: scalar as! Double) as! MatrixDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatMatrixScale(matrix as! MatrixDenseBLAS<Float>, by: scalar as! Float) as! MatrixDenseBLAS<S>
        }
        return MatrixDenseBLAS(rows: matrix.rows, columns: matrix.columns,
                               lazy: matrix.lazyMatrix.scaled(by: scalar),
                               blasImplementation: matrix.blasImplementation)
    }

    public static func * (scalar: S, matrix: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        matrix * scalar
    }

    public static func / (matrix: MatrixDenseBLAS<S>, scalar: S) -> MatrixDenseBLAS<S> {
        matrix * (1 / scalar)
    }

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
        case is Float.Type:
            let A = columnMajorStorage() as! [Float]
            let x = vectorElements(v) as! [Float]
            var y = Array(repeating: Float.zero, count: rows)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.sgemv(Int32(rows), Int32(columns), A, x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.sgemv(Int32(rows), Int32(columns), A, x, &y)
            }
            return V(y as! [S])
        case is ComplexDouble.Type:
            var A = BLASComplexStorage.interleaved(flatten() as! [ComplexDouble])
            var x = BLASComplexStorage.interleaved(v.toArray(round: false) as! [ComplexDouble])
            var y = Array(repeating: 0.0, count: rows * 2)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.zgemv(Int32(rows), Int32(columns), &A, &x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.zgemv(Int32(rows), Int32(columns), &A, &x, &y)
            }
            return V(BLASComplexStorage.complexValues(y) as! [S])
        case is ComplexFloat.Type:
            var A = BLASComplexStorage.interleaved(flatten() as! [ComplexFloat])
            var x = BLASComplexStorage.interleaved(v.toArray(round: false) as! [ComplexFloat])
            var y = Array(repeating: Float.zero, count: rows * 2)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.cgemv(Int32(rows), Int32(columns), &A, &x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.cgemv(Int32(rows), Int32(columns), &A, &x, &y)
            }
            return V(BLASComplexStorage.complexValues(y) as! [S])
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
        case is Float.Type:
            let A = columnMajorStorage() as! [Float]
            let B = columnMajorElements(from: m) as! [Float]
            var C = Array(repeating: Float.zero, count: rows * m.columns)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.sgemm(Int32(rows), Int32(m.columns), Int32(columns), A, B, &C)
            #endif
            case .openBLAS:
                OpenBLASOperations.sgemm(Int32(rows), Int32(m.columns), Int32(columns), A, B, &C)
            }
            return MatrixDenseBLAS(rows: rows, columns: m.columns, values: C as! [S])
        case is ComplexDouble.Type:
            var A = BLASComplexStorage.interleaved(flatten() as! [ComplexDouble])
            var B = BLASComplexStorage.interleaved(m.flatten() as! [ComplexDouble])
            var C = Array(repeating: 0.0, count: rows * m.columns * 2)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.zgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            #endif
            case .openBLAS:
                OpenBLASOperations.zgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            }
            return MatrixDenseBLAS(rows: rows, columns: m.columns, values: BLASComplexStorage.complexValues(C) as! [S])
        case is ComplexFloat.Type:
            var A = BLASComplexStorage.interleaved(flatten() as! [ComplexFloat])
            var B = BLASComplexStorage.interleaved(m.flatten() as! [ComplexFloat])
            var C = Array(repeating: Float.zero, count: rows * m.columns * 2)
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.cgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            #endif
            case .openBLAS:
                OpenBLASOperations.cgemm(Int32(rows), Int32(m.columns), Int32(columns), &A, &B, &C)
            }
            let values = BLASComplexStorage.complexValues(C) as! [S]
            return MatrixDenseBLAS(rows: rows, columns: m.columns, values: values)
        default:
            fatalError("Unsupported scalar type")
        }
    }
}

extension MatrixDenseBLAS {
    public var det: S {
        determinant()
    }

    @specialized(where S == Double)
    @specialized(where S == Float)
    @specialized(where S == ComplexDouble)
    @specialized(where S == ComplexFloat)
    private func determinant() -> S {
        precondition(rows == columns, "Determinant requires a square matrix")
        switch S.self {
        case is Double.Type: return doubleDeterminant() as! S
        case is Float.Type: return floatDeterminant() as! S
        case is ComplexDouble.Type: return complexDoubleDeterminant() as! S
        case is ComplexFloat.Type: return complexFloatDeterminant() as! S
        default: fatalError("Unsupported scalar type")
        }
    }

    @specialized(where S == Double)
    @specialized(where S == Float)
    @specialized(where S == ComplexDouble)
    @specialized(where S == ComplexFloat)
    public func inverse() -> MatrixDenseBLAS<S> {
        precondition(rows == columns, "Inverse requires a square matrix")
        switch S.self {
        case is Double.Type: return doubleInverse() as! MatrixDenseBLAS<S>
        case is Float.Type: return floatInverse() as! MatrixDenseBLAS<S>
        case is ComplexDouble.Type: return complexDoubleInverse() as! MatrixDenseBLAS<S>
        case is ComplexFloat.Type: return complexFloatInverse() as! MatrixDenseBLAS<S>
        default: fatalError("Unsupported scalar type")
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
}

extension MatrixDenseBLAS {
    public static func == (lhs: MatrixDenseBLAS<S>, rhs: MatrixDenseBLAS<S>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements && lhs.blasImplementation == rhs.blasImplementation
    }
}

public func + (lhs: MatrixDenseBLAS<Double>, rhs: MatrixDenseBLAS<Double>) -> MatrixDenseBLAS<Double> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return doubleMatrixSum(lhs, rhs)
}

public func + (lhs: MatrixDenseBLAS<Float>, rhs: MatrixDenseBLAS<Float>) -> MatrixDenseBLAS<Float> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return floatMatrixSum(lhs, rhs)
}

public func - (lhs: MatrixDenseBLAS<Double>, rhs: MatrixDenseBLAS<Double>) -> MatrixDenseBLAS<Double> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return doubleMatrixDifference(lhs, rhs)
}

public func - (lhs: MatrixDenseBLAS<Float>, rhs: MatrixDenseBLAS<Float>) -> MatrixDenseBLAS<Float> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return floatMatrixDifference(lhs, rhs)
}

public func * (matrix: MatrixDenseBLAS<Double>, scalar: Double) -> MatrixDenseBLAS<Double> {
    doubleMatrixScale(matrix, by: scalar)
}

public func * (scalar: Double, matrix: MatrixDenseBLAS<Double>) -> MatrixDenseBLAS<Double> {
    matrix * scalar
}

public func / (matrix: MatrixDenseBLAS<Double>, scalar: Double) -> MatrixDenseBLAS<Double> {
    matrix * (1 / scalar)
}

public func * (matrix: MatrixDenseBLAS<Float>, scalar: Float) -> MatrixDenseBLAS<Float> {
    floatMatrixScale(matrix, by: scalar)
}

public func * (scalar: Float, matrix: MatrixDenseBLAS<Float>) -> MatrixDenseBLAS<Float> {
    matrix * scalar
}

public func * (left: MatrixDenseBLAS<Double>, right: MatrixDenseBLAS<Double>) -> MatrixDenseBLAS<Double> {
    doubleMatrixProduct(left, right)
}

public func * (left: MatrixDenseBLAS<Float>, right: MatrixDenseBLAS<Float>) -> MatrixDenseBLAS<Float> {
    floatMatrixProduct(left, right)
}

public func * (left: MatrixDenseBLAS<ComplexDouble>, right: MatrixDenseBLAS<ComplexDouble>)
    -> MatrixDenseBLAS<ComplexDouble> {
    complexDoubleMatrixProduct(left, right)
}

public func * (left: MatrixDenseBLAS<ComplexFloat>, right: MatrixDenseBLAS<ComplexFloat>)
    -> MatrixDenseBLAS<ComplexFloat> {
    complexFloatMatrixProduct(left, right)
}

public func * (matrix: MatrixDenseBLAS<Double>, vector: VectorDenseBLAS<Double>) -> VectorDenseBLAS<Double> {
    doubleMatrixVectorProduct(matrix, vector)
}

public func * (matrix: MatrixDenseBLAS<Float>, vector: VectorDenseBLAS<Float>) -> VectorDenseBLAS<Float> {
    floatMatrixVectorProduct(matrix, vector)
}

public func * (matrix: MatrixDenseBLAS<ComplexDouble>, vector: VectorDenseBLAS<ComplexDouble>)
    -> VectorDenseBLAS<ComplexDouble> {
    complexDoubleMatrixVectorProduct(matrix, vector)
}

public func * (matrix: MatrixDenseBLAS<ComplexFloat>, vector: VectorDenseBLAS<ComplexFloat>)
    -> VectorDenseBLAS<ComplexFloat> {
    complexFloatMatrixVectorProduct(matrix, vector)
}

public func / (matrix: MatrixDenseBLAS<Float>, scalar: Float) -> MatrixDenseBLAS<Float> {
    matrix * (1 / scalar)
}

private func doubleMatrixSum(
    _ lhs: MatrixDenseBLAS<Double>, _ rhs: MatrixDenseBLAS<Double>
) -> MatrixDenseBLAS<Double> {
    if lhs.isWholeMaterializedMatrix && rhs.isWholeMaterializedMatrix {
        return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                               values: doubleSum(lhs.view.storage.elements, rhs.view.storage.elements),
                               blasImplementation: lhs.blasImplementation)
    }
    return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                           lazy: lhs.lazyMatrix.adding(rhs.lazyMatrix),
                           blasImplementation: lhs.blasImplementation)
}

private func floatMatrixSum(_ lhs: MatrixDenseBLAS<Float>, _ rhs: MatrixDenseBLAS<Float>) -> MatrixDenseBLAS<Float> {
    if lhs.isWholeMaterializedMatrix && rhs.isWholeMaterializedMatrix {
        return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                               values: floatSum(lhs.view.storage.elements, rhs.view.storage.elements),
                               blasImplementation: lhs.blasImplementation)
    }
    return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                           lazy: lhs.lazyMatrix.adding(rhs.lazyMatrix),
                           blasImplementation: lhs.blasImplementation)
}

private func doubleMatrixDifference(
    _ lhs: MatrixDenseBLAS<Double>, _ rhs: MatrixDenseBLAS<Double>
) -> MatrixDenseBLAS<Double> {
    if lhs.isWholeMaterializedMatrix && rhs.isWholeMaterializedMatrix {
        return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                               values: doubleDifference(lhs.view.storage.elements, rhs.view.storage.elements),
                               blasImplementation: lhs.blasImplementation)
    }
    return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                           lazy: lhs.lazyMatrix.subtracting(rhs.lazyMatrix),
                           blasImplementation: lhs.blasImplementation)
}

private func floatMatrixDifference(
    _ lhs: MatrixDenseBLAS<Float>, _ rhs: MatrixDenseBLAS<Float>
) -> MatrixDenseBLAS<Float> {
    if lhs.isWholeMaterializedMatrix && rhs.isWholeMaterializedMatrix {
        return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                               values: floatDifference(lhs.view.storage.elements, rhs.view.storage.elements),
                               blasImplementation: lhs.blasImplementation)
    }
    return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns,
                           lazy: lhs.lazyMatrix.subtracting(rhs.lazyMatrix),
                           blasImplementation: lhs.blasImplementation)
}

private func doubleMatrixScale(_ matrix: MatrixDenseBLAS<Double>, by scalar: Double) -> MatrixDenseBLAS<Double> {
    if matrix.isWholeMaterializedMatrix {
        return MatrixDenseBLAS(rows: matrix.rows, columns: matrix.columns,
                               values: doubleScale(matrix.view.storage.elements, by: scalar),
                               blasImplementation: matrix.blasImplementation)
    }
    return MatrixDenseBLAS(rows: matrix.rows, columns: matrix.columns,
                           lazy: matrix.lazyMatrix.scaled(by: scalar),
                           blasImplementation: matrix.blasImplementation)
}

private func floatMatrixScale(_ matrix: MatrixDenseBLAS<Float>, by scalar: Float) -> MatrixDenseBLAS<Float> {
    if matrix.isWholeMaterializedMatrix {
        return MatrixDenseBLAS(rows: matrix.rows, columns: matrix.columns,
                               values: floatScale(matrix.view.storage.elements, by: scalar),
                               blasImplementation: matrix.blasImplementation)
    }
    return MatrixDenseBLAS(rows: matrix.rows, columns: matrix.columns,
                           lazy: matrix.lazyMatrix.scaled(by: scalar),
                           blasImplementation: matrix.blasImplementation)
}

private func doubleMatrixProduct(
    _ left: MatrixDenseBLAS<Double>, _ right: MatrixDenseBLAS<Double>
) -> MatrixDenseBLAS<Double> {
    precondition(left.columns == right.rows, "Number of matrix columns must match matrix rows")
    let leftElements = left.columnMajorStorage()
    let rightElements = right.columnMajorStorage()
    var result = Array(repeating: 0.0, count: left.rows * right.columns)
    switch left.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        AccelerateOperations.dgemm(Int32(left.rows), Int32(right.columns), Int32(left.columns),
                                   leftElements, rightElements, &result)
    #endif
    case .openBLAS:
        OpenBLASOperations.dgemm(Int32(left.rows), Int32(right.columns), Int32(left.columns),
                                 leftElements, rightElements, &result)
    }
    return MatrixDenseBLAS<Double>(rows: left.rows, columns: right.columns, values: result,
                                   blasImplementation: left.blasImplementation)
}

private func floatMatrixProduct(_ left: MatrixDenseBLAS<Float>, _ right: MatrixDenseBLAS<Float>)
    -> MatrixDenseBLAS<Float> {
    precondition(left.columns == right.rows, "Number of matrix columns must match matrix rows")
    let leftElements = left.columnMajorStorage()
    let rightElements = right.columnMajorStorage()
    var result = Array(repeating: Float.zero, count: left.rows * right.columns)
    switch left.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        AccelerateOperations.sgemm(Int32(left.rows), Int32(right.columns), Int32(left.columns),
                                   leftElements, rightElements, &result)
    #endif
    case .openBLAS:
        OpenBLASOperations.sgemm(Int32(left.rows), Int32(right.columns), Int32(left.columns),
                                 leftElements, rightElements, &result)
    }
    return MatrixDenseBLAS<Float>(rows: left.rows, columns: right.columns, values: result,
                                  blasImplementation: left.blasImplementation)
}

private func complexDoubleMatrixProduct(
    _ left: MatrixDenseBLAS<ComplexDouble>, _ right: MatrixDenseBLAS<ComplexDouble>
) -> MatrixDenseBLAS<ComplexDouble> {
    precondition(left.columns == right.rows, "Number of matrix columns must match matrix rows")
    let leftElements = left.columnMajorStorage()
    let rightElements = right.columnMajorStorage()
    var result = Array(repeating: ComplexDouble.zero, count: left.rows * right.columns)
    switch left.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        complexDoubleGEMM(left.rows, right.columns, left.columns, leftElements, rightElements, &result,
                          AccelerateOperations.zgemmRaw)
    #endif
    case .openBLAS:
        complexDoubleGEMM(left.rows, right.columns, left.columns, leftElements, rightElements, &result,
                          OpenBLASOperations.zgemmRaw)
    }
    return MatrixDenseBLAS<ComplexDouble>(rows: left.rows, columns: right.columns, values: result,
                                          blasImplementation: left.blasImplementation)
}

private func complexFloatMatrixProduct(
    _ left: MatrixDenseBLAS<ComplexFloat>, _ right: MatrixDenseBLAS<ComplexFloat>
) -> MatrixDenseBLAS<ComplexFloat> {
    precondition(left.columns == right.rows, "Number of matrix columns must match matrix rows")
    let leftElements = left.columnMajorStorage()
    let rightElements = right.columnMajorStorage()
    var result = Array(repeating: ComplexFloat.zero, count: left.rows * right.columns)
    switch left.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        complexFloatGEMM(left.rows, right.columns, left.columns, leftElements, rightElements, &result,
                         AccelerateOperations.cgemmRaw)
    #endif
    case .openBLAS:
        complexFloatGEMM(left.rows, right.columns, left.columns, leftElements, rightElements, &result,
                         OpenBLASOperations.cgemmRaw)
    }
    return MatrixDenseBLAS<ComplexFloat>(rows: left.rows, columns: right.columns, values: result,
                                         blasImplementation: left.blasImplementation)
}

private func doubleMatrixVectorProduct(
    _ matrix: MatrixDenseBLAS<Double>, _ vector: VectorDenseBLAS<Double>
) -> VectorDenseBLAS<Double> {
    precondition(matrix.columns == vector.size, "Number of columns in matrix must equal size of vector")
    let matrixElements = matrix.columnMajorStorage()
    let vectorElements = vector.elements
    var result = Array(repeating: 0.0, count: matrix.rows)
    switch matrix.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        AccelerateOperations.dgemv(Int32(matrix.rows), Int32(matrix.columns), matrixElements, vectorElements, &result)
    #endif
    case .openBLAS:
        OpenBLASOperations.dgemv(Int32(matrix.rows), Int32(matrix.columns), matrixElements, vectorElements, &result)
    }
    return VectorDenseBLAS<Double>(result)
}

private func floatMatrixVectorProduct(
    _ matrix: MatrixDenseBLAS<Float>, _ vector: VectorDenseBLAS<Float>
) -> VectorDenseBLAS<Float> {
    precondition(matrix.columns == vector.size, "Number of columns in matrix must equal size of vector")
    let matrixElements = matrix.columnMajorStorage()
    let vectorElements = vector.elements
    var result = Array(repeating: Float.zero, count: matrix.rows)
    switch matrix.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        AccelerateOperations.sgemv(Int32(matrix.rows), Int32(matrix.columns), matrixElements, vectorElements, &result)
    #endif
    case .openBLAS:
        OpenBLASOperations.sgemv(Int32(matrix.rows), Int32(matrix.columns), matrixElements, vectorElements, &result)
    }
    return VectorDenseBLAS<Float>(result)
}

private func complexDoubleMatrixVectorProduct(
    _ matrix: MatrixDenseBLAS<ComplexDouble>, _ vector: VectorDenseBLAS<ComplexDouble>
) -> VectorDenseBLAS<ComplexDouble> {
    precondition(matrix.columns == vector.size, "Number of columns in matrix must equal size of vector")
    let matrixElements = matrix.columnMajorStorage()
    let vectorElements = vector.elements
    var result = Array(repeating: ComplexDouble.zero, count: matrix.rows)
    switch matrix.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        complexDoubleGEMV(matrix.rows, matrix.columns, matrixElements, vectorElements, &result,
                          AccelerateOperations.zgemvRaw)
    #endif
    case .openBLAS:
        complexDoubleGEMV(matrix.rows, matrix.columns, matrixElements, vectorElements, &result,
                          OpenBLASOperations.zgemvRaw)
    }
    return VectorDenseBLAS<ComplexDouble>(result)
}

private func complexFloatMatrixVectorProduct(
    _ matrix: MatrixDenseBLAS<ComplexFloat>, _ vector: VectorDenseBLAS<ComplexFloat>
) -> VectorDenseBLAS<ComplexFloat> {
    precondition(matrix.columns == vector.size, "Number of columns in matrix must equal size of vector")
    let matrixElements = matrix.columnMajorStorage()
    let vectorElements = vector.elements
    var result = Array(repeating: ComplexFloat.zero, count: matrix.rows)
    switch matrix.blasImplementation {
    #if canImport(Accelerate)
    case .accelerate:
        complexFloatGEMV(matrix.rows, matrix.columns, matrixElements, vectorElements, &result,
                         AccelerateOperations.cgemvRaw)
    #endif
    case .openBLAS:
        complexFloatGEMV(matrix.rows, matrix.columns, matrixElements, vectorElements, &result,
                         OpenBLASOperations.cgemvRaw)
    }
    return VectorDenseBLAS<ComplexFloat>(result)
}

private func complexDoubleGEMM(
    _ m: Int, _ n: Int, _ k: Int, _ a: [ComplexDouble], _ b: [ComplexDouble], _ c: inout [ComplexDouble],
    _ gemm: (Int32, Int32, Int32, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer) -> Void
) {
    BLASComplexStorage.withUnsafeInterleavedStorage(a) { a in
        BLASComplexStorage.withUnsafeInterleavedStorage(b) { b in
            BLASComplexStorage.withUnsafeMutableInterleavedStorage(&c) { c in
                gemm(Int32(m), Int32(n), Int32(k), a, b, c)
            }
        }
    }
}

private func complexFloatGEMM(
    _ m: Int, _ n: Int, _ k: Int, _ a: [ComplexFloat], _ b: [ComplexFloat], _ c: inout [ComplexFloat],
    _ gemm: (Int32, Int32, Int32, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer) -> Void
) {
    BLASComplexStorage.withUnsafeInterleavedStorage(a) { a in
        BLASComplexStorage.withUnsafeInterleavedStorage(b) { b in
            BLASComplexStorage.withUnsafeMutableInterleavedStorage(&c) { c in
                gemm(Int32(m), Int32(n), Int32(k), a, b, c)
            }
        }
    }
}

private func complexDoubleGEMV(
    _ m: Int, _ n: Int, _ a: [ComplexDouble], _ x: [ComplexDouble], _ y: inout [ComplexDouble],
    _ gemv: (Int32, Int32, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer) -> Void
) {
    BLASComplexStorage.withUnsafeInterleavedStorage(a) { a in
        BLASComplexStorage.withUnsafeInterleavedStorage(x) { x in
            BLASComplexStorage.withUnsafeMutableInterleavedStorage(&y) { y in gemv(Int32(m), Int32(n), a, x, y) }
        }
    }
}

private func complexFloatGEMV(
    _ m: Int, _ n: Int, _ a: [ComplexFloat], _ x: [ComplexFloat], _ y: inout [ComplexFloat],
    _ gemv: (Int32, Int32, UnsafeRawPointer, UnsafeRawPointer, UnsafeMutableRawPointer) -> Void
) {
    BLASComplexStorage.withUnsafeInterleavedStorage(a) { a in
        BLASComplexStorage.withUnsafeInterleavedStorage(x) { x in
            BLASComplexStorage.withUnsafeMutableInterleavedStorage(&y) { y in gemv(Int32(m), Int32(n), a, x, y) }
        }
    }
}

extension MatrixDenseBLAS {
    fileprivate var lazyMatrix: LazyMatrix<S> { lazy ?? .view(view) }
    fileprivate var isWholeMaterializedMatrix: Bool {
        lazy == nil && view.offset == 0 && view.isContiguous && view.storage.elements.count == rows * columns
    }

    private func sum(_ left: [S], _ right: [S]) -> [S] {
        switch S.self {
        case is Double.Type:
            let x = right as! [Double]
            var y = left as! [Double]
            axpy(Int32(y.count), x, &y)
            return y as! [S]
        case is Float.Type:
            let x = right as! [Float]
            var y = left as! [Float]
            axpy(Int32(y.count), x, &y)
            return y as! [S]
        case is ComplexDouble.Type:
            return BLASComplexStorage.sum(left as! [ComplexDouble], right as! [ComplexDouble]) as! [S]
        case is ComplexFloat.Type:
            return BLASComplexStorage.sum(left as! [ComplexFloat], right as! [ComplexFloat]) as! [S]
        default:
            fatalError("Unsupported scalar type")
        }
    }

    private func axpy(_ n: Int32, _ x: [Double], _ y: inout [Double]) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: AccelerateOperations.daxpy(n, x, &y)
        #endif
        case .openBLAS: OpenBLASOperations.daxpy(n, x, &y)
        }
    }

    private func axpy(_ n: Int32, _ x: [Float], _ y: inout [Float]) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: AccelerateOperations.saxpy(n, x, &y)
        #endif
        case .openBLAS: OpenBLASOperations.saxpy(n, x, &y)
        }
    }

    private func zaxpy(_ n: Int32, _ x: inout [Double], _ y: inout [Double]) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: AccelerateOperations.zaxpy(n, &x, &y)
        #endif
        case .openBLAS: OpenBLASOperations.zaxpy(n, &x, &y)
        }
    }

    private func caxpy(_ n: Int32, _ x: inout [Float], _ y: inout [Float]) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: AccelerateOperations.caxpy(n, &x, &y)
        #endif
        case .openBLAS: OpenBLASOperations.caxpy(n, &x, &y)
        }
    }

}

private func doubleSum(_ left: [Double], _ right: [Double]) -> [Double] {
    Array<Double>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] + right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func floatSum(_ left: [Float], _ right: [Float]) -> [Float] {
    Array<Float>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] + right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func doubleDifference(_ left: [Double], _ right: [Double]) -> [Double] {
    Array<Double>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] - right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func floatDifference(_ left: [Float], _ right: [Float]) -> [Float] {
    Array<Float>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] - right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func doubleScale(_ values: [Double], by scalar: Double) -> [Double] {
    Array<Double>(unsafeUninitializedCapacity: values.count) { result, initializedCount in
        values.withUnsafeBufferPointer { values in
            for index in 0..<values.count { result[index] = values[index] * scalar }
        }
        initializedCount = values.count
    }
}

private func floatScale(_ values: [Float], by scalar: Float) -> [Float] {
    Array<Float>(unsafeUninitializedCapacity: values.count) { result, initializedCount in
        values.withUnsafeBufferPointer { values in
            for index in 0..<values.count { result[index] = values[index] * scalar }
        }
        initializedCount = values.count
    }
}

extension MatrixDenseBLAS {
    private func value(row: Int, column: Int) -> S {
        if let lazy { return lazy.value(row: row, column: column) }
        return view.value(index0: row, index1: column)
    }

    private mutating func setValue(_ value: S, row: Int, column: Int) {
        materializeInPlace()
        view.setValue(value, index0: row, index1: column)
    }

    private mutating func assign(_ replacement: MatrixDenseBLAS<S>, to ranges: [SliceRange]) {
        materializeInPlace()
        if let lazy = replacement.lazy {
            view.assign(lazy, rows: replacement.rows, columns: replacement.columns, to: ranges)
        } else {
            view.assign(replacement.view, to: ranges)
        }
    }

    private func flattenedFromView(columnMajorOrder: Bool) -> [S] {
        var elements = Array(repeating: S.zero, count: rows * columns)
        if columnMajorOrder {
            for column in 0..<columns {
                for row in 0..<rows {
                    let index = row + rows * column
                    let storageIndex = view.offset + row * view.strides[0] + column * view.strides[1]
                    elements[index] = view.storage.elements[storageIndex]
                }
            }
        } else {
            for row in 0..<rows {
                for column in 0..<columns {
                    let index = column + columns * row
                    let storageIndex = view.offset + row * view.strides[0] + column * view.strides[1]
                    elements[index] = view.storage.elements[storageIndex]
                }
            }
        }
        return elements
    }

    private func rowMajorElements(fromColumnMajorElements elements: [S]) -> [S] {
        var rowMajorElements = Array(repeating: S.zero, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns { rowMajorElements[column + columns * row] = elements[row + rows * column] }
        }
        return rowMajorElements
    }

    fileprivate func columnMajorStorage() -> [S] {
        if let lazy { return lazy.materializedElements(rows: rows, columns: columns) }
        return view.contiguousElements ?? flattenedFromView(columnMajorOrder: true)
    }

    private func materializedView() -> TensorFlatView<S> {
        if let lazy {
            return TensorFlatView(shape: shape, elements: lazy.materializedElements(rows: rows, columns: columns))
        }
        return view
    }

    private mutating func materializeInPlace() {
        guard let lazy else { return }
        view = TensorFlatView(shape: shape, elements: lazy.materializedElements(rows: rows, columns: columns))
        self.lazy = nil
    }

    var isWholeContiguousView: Bool { lazy == nil && view.isContiguous && view.offset == 0 }

    private func vectorElements<V: PluVector>(_ vector: V) -> [S] where V.S == S {
        if let vector = vector as? VectorDenseBLAS<S> { return vector.elements }
        return vector.toArray(round: false)
    }

    private func columnMajorElements<M: PluMatrix>(from matrix: M) -> [S] where M.S == S {
        if let matrix = matrix as? MatrixDenseBLAS<S> { return matrix.columnMajorStorage() }
        return matrix.flatten(columnMajorOrder: true)
    }

    private func doubleFactorization() -> (matrix: [Double], pivots: [Int32], info: Int32) {
        var matrix = columnMajorStorage() as! [Double]
        let factorization = doubleGetrf(Int32(rows), &matrix)
        return (matrix, factorization.pivots, factorization.info)
    }

    private func floatFactorization() -> (matrix: [Float], pivots: [Int32], info: Int32) {
        var matrix = columnMajorStorage() as! [Float]
        let factorization = floatGetrf(Int32(rows), &matrix)
        return (matrix, factorization.pivots, factorization.info)
    }

    private func complexDoubleFactorization() -> (matrix: [Double], pivots: [Int32], info: Int32) {
        var matrix = BLASComplexStorage.interleaved(columnMajorStorage() as! [ComplexDouble])
        let factorization = complexDoubleGetrf(Int32(rows), &matrix)
        return (matrix, factorization.pivots, factorization.info)
    }

    private func complexFloatFactorization() -> (matrix: [Float], pivots: [Int32], info: Int32) {
        var matrix = BLASComplexStorage.interleaved(columnMajorStorage() as! [ComplexFloat])
        let factorization = complexFloatGetrf(Int32(rows), &matrix)
        return (matrix, factorization.pivots, factorization.info)
    }

    private func doubleDeterminant() -> Double {
        let factorization = doubleFactorization()
        precondition(factorization.info >= 0, "LAPACK determinant failed with info \(factorization.info)")
        if factorization.info > 0 { return .zero }
        return luDeterminant(from: factorization.matrix, pivots: factorization.pivots, one: 1.0)
    }

    private func floatDeterminant() -> Float {
        let factorization = floatFactorization()
        precondition(factorization.info >= 0, "LAPACK determinant failed with info \(factorization.info)")
        if factorization.info > 0 { return .zero }
        return luDeterminant(from: factorization.matrix, pivots: factorization.pivots, one: Float(1.0))
    }

    private func complexDoubleDeterminant() -> ComplexDouble {
        let factorization = complexDoubleFactorization()
        precondition(factorization.info >= 0, "LAPACK determinant failed with info \(factorization.info)")
        if factorization.info > 0 { return .zero }
        let matrix = BLASComplexStorage.complexValues(factorization.matrix) as [ComplexDouble]
        return luDeterminant(from: matrix, pivots: factorization.pivots, one: ComplexDouble(1.0, 0.0))
    }

    private func complexFloatDeterminant() -> ComplexFloat {
        let factorization = complexFloatFactorization()
        precondition(factorization.info >= 0, "LAPACK determinant failed with info \(factorization.info)")
        if factorization.info > 0 { return .zero }
        let matrix = BLASComplexStorage.complexValues(factorization.matrix) as [ComplexFloat]
        return luDeterminant(from: matrix, pivots: factorization.pivots, one: ComplexFloat(1.0, 0.0))
    }

    private func luDeterminant<T: PluScalar>(from matrix: [T], pivots: [Int32], one: T) -> T {
        var result = one
        for index in 0..<rows {
            if pivots[index] != Int32(index + 1) { result = -result }
            result *= matrix[index + rows * index]
        }
        return result
    }

    private func doubleInverse() -> MatrixDenseBLAS<Double> {
        var factorization = doubleFactorization()
        precondition(factorization.info == 0, "Matrix must be invertible")
        let info = doubleGetri(Int32(rows), &factorization.matrix, factorization.pivots)
        precondition(info == 0, "LAPACK inverse failed with info \(info)")
        return MatrixDenseBLAS<Double>(rows: rows, columns: columns, values: factorization.matrix)
    }

    private func floatInverse() -> MatrixDenseBLAS<Float> {
        var factorization = floatFactorization()
        precondition(factorization.info == 0, "Matrix must be invertible")
        let info = floatGetri(Int32(rows), &factorization.matrix, factorization.pivots)
        precondition(info == 0, "LAPACK inverse failed with info \(info)")
        return MatrixDenseBLAS<Float>(rows: rows, columns: columns, values: factorization.matrix)
    }

    private func complexDoubleInverse() -> MatrixDenseBLAS<ComplexDouble> {
        var factorization = complexDoubleFactorization()
        precondition(factorization.info == 0, "Matrix must be invertible")
        let info = complexDoubleGetri(Int32(rows), &factorization.matrix, factorization.pivots)
        precondition(info == 0, "LAPACK inverse failed with info \(info)")
        return MatrixDenseBLAS<ComplexDouble>(rows: rows, columns: columns,
                                              values: BLASComplexStorage.complexValues(factorization.matrix))
    }

    private func complexFloatInverse() -> MatrixDenseBLAS<ComplexFloat> {
        var factorization = complexFloatFactorization()
        precondition(factorization.info == 0, "Matrix must be invertible")
        let info = complexFloatGetri(Int32(rows), &factorization.matrix, factorization.pivots)
        precondition(info == 0, "LAPACK inverse failed with info \(info)")
        return MatrixDenseBLAS<ComplexFloat>(rows: rows, columns: columns,
                                             values: BLASComplexStorage.complexValues(factorization.matrix))
    }

    private func doubleGetrf(_ n: Int32, _ matrix: inout [Double]) -> (pivots: [Int32], info: Int32) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.dgetrf(n, &matrix)
        #endif
        case .openBLAS: return OpenBLASOperations.dgetrf(n, &matrix)
        }
    }

    private func floatGetrf(_ n: Int32, _ matrix: inout [Float]) -> (pivots: [Int32], info: Int32) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.sgetrf(n, &matrix)
        #endif
        case .openBLAS: return OpenBLASOperations.sgetrf(n, &matrix)
        }
    }

    private func complexDoubleGetrf(_ n: Int32, _ matrix: inout [Double]) -> (pivots: [Int32], info: Int32) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.zgetrf(n, &matrix)
        #endif
        case .openBLAS: return OpenBLASOperations.zgetrf(n, &matrix)
        }
    }

    private func complexFloatGetrf(_ n: Int32, _ matrix: inout [Float]) -> (pivots: [Int32], info: Int32) {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.cgetrf(n, &matrix)
        #endif
        case .openBLAS: return OpenBLASOperations.cgetrf(n, &matrix)
        }
    }

    private func doubleGetri(_ n: Int32, _ matrix: inout [Double], _ pivots: [Int32]) -> Int32 {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.dgetri(n, &matrix, pivots)
        #endif
        case .openBLAS: return OpenBLASOperations.dgetri(n, &matrix, pivots)
        }
    }

    private func floatGetri(_ n: Int32, _ matrix: inout [Float], _ pivots: [Int32]) -> Int32 {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.sgetri(n, &matrix, pivots)
        #endif
        case .openBLAS: return OpenBLASOperations.sgetri(n, &matrix, pivots)
        }
    }

    private func complexDoubleGetri(_ n: Int32, _ matrix: inout [Double], _ pivots: [Int32]) -> Int32 {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.zgetri(n, &matrix, pivots)
        #endif
        case .openBLAS: return OpenBLASOperations.zgetri(n, &matrix, pivots)
        }
    }

    private func complexFloatGetri(_ n: Int32, _ matrix: inout [Float], _ pivots: [Int32]) -> Int32 {
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: return AccelerateOperations.cgetri(n, &matrix, pivots)
        #endif
        case .openBLAS: return OpenBLASOperations.cgetri(n, &matrix, pivots)
        }
    }
}

extension MatrixDenseBLAS: MatrixEigen where S == Double {
    public typealias Eigenvalue = ComplexDouble
    public typealias Eigenvectors = MatrixDenseBLAS<ComplexDouble>

    public func eigen() -> Eigen<ComplexDouble, MatrixDenseBLAS<ComplexDouble>> {
        precondition(rows == columns, "Eigen decomposition requires a square matrix")
        let n = Int32(rows)
        var matrix = flatten()
        var real = Array(repeating: 0.0, count: rows)
        var imaginary = Array(repeating: 0.0, count: rows)
        var vectors = Array(repeating: 0.0, count: rows * columns)
        let info: Int32
        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate: info = AccelerateOperations.dgeev(n, &matrix, &real, &imaginary, &vectors)
        #endif
        case .openBLAS: info = OpenBLASOperations.dgeev(n, &matrix, &real, &imaginary, &vectors)
        }
        precondition(info == 0, "Eigen decomposition failed with LAPACK info \(info)")
        return Eigen(values: eigenvalues(real: real, imaginary: imaginary),
                     vectors: eigenvectors(real: real, imaginary: imaginary, vectors: vectors))
    }

    private func eigenvalues(real: [Double], imaginary: [Double]) -> [ComplexDouble] {
        zip(real, imaginary).map { ComplexDouble($0, $1) }
    }

    private func eigenvectors(
        real: [Double], imaginary: [Double], vectors: [Double]
    ) -> MatrixDenseBLAS<ComplexDouble> {
        var result = MatrixDenseBLAS<ComplexDouble>(rows: rows, columns: columns, initialValue: .zero)
        var column = 0
        while column < columns {
            if imaginary[column] == 0.0 {
                for row in 0..<rows {
                    result[row, column] = ComplexDouble(vectors[row + rows * column], 0.0)
                }
                column += 1
            } else {
                for row in 0..<rows {
                    let realPart = vectors[row + rows * column]
                    let imaginaryPart = vectors[row + rows * (column + 1)]
                    result[row, column] = ComplexDouble(realPart, imaginaryPart)
                    result[row, column + 1] = ComplexDouble(realPart, -imaginaryPart)
                }
                column += 2
            }
        }
        return result
    }
}
