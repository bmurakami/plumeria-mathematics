import AccelerateWrapper
import OpenBLASWrapper

public struct MatrixDenseBLAS<S: PluScalar>: PluMatrix {
    private var values: [S]
    private var n_r: Int
    private var n_c: Int
    public var blasImplementation: BLAS
    
    private func index(row: Int, column: Int) -> Int {
        row + n_r * column
    }
    
    private func value(row: Int, column: Int) -> S {
        values[index(row: row, column: column)]
    }
    
    private mutating func setValue(_ value: S, row: Int, column: Int) {
        values[index(row: row, column: column)] = value
    }

    init(rows: Int, columns: Int, values: [S], blasImplementation: BLAS = BLAS.default) {
        self.n_r = rows
        self.n_c = columns
        self.values = values
        self.blasImplementation = blasImplementation
    }
    
    // MARK: - PluMatrix conformance
    public var rows: Int { return n_r }
    public var columns: Int { return n_c }
    
    public subscript(i: Int, j: Int) -> S {
        get { value(row: i, column: j) }
        set { setValue(newValue, row: i, column: j) }
    }
    
    public init(rows: Int, columns: Int, initialValue: S = S.zero) {
        self.n_r = rows
        self.n_c = columns
        self.values = Array(repeating: initialValue, count: rows * columns)
        self.blasImplementation = BLAS.default
    }

    public init(_ values: [[S]]) {
        func storeAsColumnMajor(_ values: [[S]]) {
            self.values = (0..<n_c).flatMap { j in
                (0..<n_r).map { i in
                    values[i][j]
                }
            }
        }
        
        self.n_r = values.count
        self.n_c = values[0].count
        self.values = [S.zero]
        self.blasImplementation = .openBLAS
        storeAsColumnMajor(values)
    }

    public var elements: [S] {
        get { values }
        set {
            precondition(newValue.count == rows * columns, "Matrix element count must match matrix shape")
            values = newValue
        }
    }
    
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { shape.count }
    
    func view() -> TensorView<S> {
        TensorView(storage: TensorStorage(values), offset: 0, shape: shape, strides: [1, rows])
    }
    
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

    public func times<V: PluVector>(_ v: V) -> V where V.S == S {
        precondition(n_c == v.size, "Number of columns in matrix must equal size of vector")
        
        var y = Array(repeating: 0.0, count: v.size)
        
        switch S.self {
        case is Double.Type:
            var A = flatten() as! [Double] // Ax = y
            var x = v.toArray(round: false) as! [Double]
            
            switch blasImplementation {
            #if canImport(Accelerate)
            case .accelerate:
                AccelerateOperations.dgemv(Int32(n_r), Int32(n_c), &A, &x, &y)
            #endif
            case .openBLAS:
                OpenBLASOperations.dgemv(Int32(n_r), Int32(n_c), &A, &x, &y)
            }
        case is Complex.Type:
            fatalError("Not yet implemented")
        default:
            fatalError("Unsupported scalar type")
        }
        
        return V(y as! [S])
    }
    
    public func transpose() -> MatrixDenseBLAS<S> {
        var mt = MatrixDenseBLAS(rows: n_c, columns: n_r)
        for i in 0..<n_r {
            for j in 0..<n_c {
                mt.setValue(value(row: i, column: j), row: j, column: i)
            }
        }
        return mt
    }
 
    public func toArray(round: Bool = false) -> [[S]] {
        return (0..<n_r).map { i in
                    (0..<n_c).map { j in
                        let value = value(row: i, column: j)
                        return round ? value.round() : value
                    }
                }
    }
    
    public func flatten(columnMajorOrder: Bool = true) -> [S] {
        if columnMajorOrder {
            return values
        } else {
            var flattened = Array(repeating: S.zero, count: n_r * n_c)
            for i in 0..<n_r {
                for j in 0..<n_c {
                    flattened[j + n_c * i] = value(row: i, column: j)
                }
            }
            return flattened
        }
    }
    
    // MARK: - PluTensor conformance
    public static func + (lhs: MatrixDenseBLAS<S>, rhs: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        precondition(lhs.n_r == rhs.n_r && lhs.n_c == rhs.n_c, "Matrices must have the same shape.")
        
        return MatrixDenseBLAS(rows: lhs.n_r, columns: lhs.n_c, values: zip(lhs.values, rhs.values).map { $0 + $1 })
    }
    
    public static prefix func - (operand: MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S> {
        return MatrixDenseBLAS(rows: operand.n_r, columns: operand.n_c, values: operand.values.map { -$0 })
    }
}
