import AccelerateWrapper
import Numerics
import OpenBLASWrapper

public struct MatrixDenseBLAS<S: PluScalar>: TensorArithmeticBLAS, MatrixColumnMajorInitializable, Equatable {
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
        precondition(rows == columns, "Determinant requires a square matrix")
        switch S.self {
        case is Double.Type: return doubleDeterminant() as! S
        case is Float.Type: return floatDeterminant() as! S
        case is ComplexDouble.Type: return complexDoubleDeterminant() as! S
        case is ComplexFloat.Type: return complexFloatDeterminant() as! S
        default: fatalError("Unsupported scalar type")
        }
    }

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
