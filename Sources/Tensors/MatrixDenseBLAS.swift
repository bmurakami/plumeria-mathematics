import AccelerateWrapper
import OpenBLASWrapper

public struct MatrixDenseBLAS<S: PluScalar>: PluMatrix {
    private var view: TensorView<S>
    public var blasImplementation: BLAS
    
    private func value(row: Int, column: Int) -> S {
        view[[row, column]]
    }
    
    private mutating func setValue(_ value: S, row: Int, column: Int) {
        view[[row, column]] = value
    }

    init(rows: Int, columns: Int, values: [S], blasImplementation: BLAS = BLAS.default) {
        self.view = TensorView(shape: [rows, columns], elements: values)
        self.blasImplementation = blasImplementation
    }
    
    init(view: TensorView<S>, blasImplementation: BLAS = BLAS.default) {
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
        self.view = TensorView(shape: [rows, columns], elements: elements)
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
        self.view = TensorView(shape: [rows, columns], elements: elements)
        self.blasImplementation = .openBLAS
    }

    public var elements: [S] {
        get { viewElements(columnMajorOrder: true) }
        set {
            precondition(newValue.count == rows * columns, "Matrix element count must match matrix shape")
            view = TensorView(shape: shape, elements: newValue)
        }
    }
    
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { shape.count }
    
    public init(shape: [Int]) {
        precondition(shape.count == 2, "MatrixDenseBLAS shape must have rank 2")
        precondition(shape.allSatisfy { $0 >= 0 }, "Matrix shape dimensions must be non-negative")
        
        self.init(rows: shape[0], columns: shape[1])
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

    public func times<V: PluVector>(_ v: V) -> V where V.S == S {
        precondition(columns == v.size, "Number of columns in matrix must equal size of vector")
        
        var y = Array(repeating: 0.0, count: rows)
        
        switch S.self {
        case is Double.Type:
            var A = flatten() as! [Double] // Ax = y
            var x = v.toArray(round: false) as! [Double]
            
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.dgemv(Int32(rows), Int32(columns), &A, &x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.dgemv(Int32(rows), Int32(columns), &A, &x, &y)
            }
        case is Complex.Type:
            fatalError("Not yet implemented")
        default:
            fatalError("Unsupported scalar type")
        }
        
        return V(y as! [S])
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
    
    // MARK: - PluTensor conformance
    public static func + (lhs: MatrixDenseBLAS<S>, rhs: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Matrices must have the same shape.")
        
        let elements = zip(lhs.elements, rhs.elements).map { $0 + $1 }
        return MatrixDenseBLAS(rows: lhs.rows, columns: lhs.columns, values: elements)
    }
    
    public static prefix func - (operand: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        return MatrixDenseBLAS(rows: operand.rows, columns: operand.columns, values: operand.elements.map { -$0 })
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
}
