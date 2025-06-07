import OpenBLASWrapper

public struct MatrixDenseBLAS<S: PluScalar>: PluMatrix  {
    private var values: [S]
    private var n_r: Int
    private var n_c: Int
    public var blasImplementation: BLASImplementation

    init(rows: Int, columns: Int, values: [S], blasImplementation: BLASImplementation = .openBLAS) {
        self.n_r = rows
        self.n_c = columns
        self.values = values
        self.blasImplementation = blasImplementation
    }
    
    // MARK: - PluMatrix conformance
    public var rows: Int { return n_r }
    public var columns: Int { return n_c }

    public var transpose: MatrixDenseBLAS<S> {
        var mt = MatrixDenseBLAS(rows: n_c, columns: n_r)
        for i in 0..<n_r {
            for j in 0..<n_c {
                mt[j, i] = values[i + n_r * j]
            }
        }
        return mt
    }
    
    public subscript(i: Int, j: Int) -> S {
        get { values[i + n_r * j] }
        set { values[i + n_r * j] = newValue }
    }
    
    public init(rows: Int, columns: Int, initialValue: S = S.zero) {
        self.n_r = rows
        self.n_c = columns
        self.values = Array(repeating: initialValue, count: rows * columns)
        self.blasImplementation = .openBLAS
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

    public func times<V: PluVector>(_ v: V) -> V where V.S == S {
        precondition(n_c == v.size, "Number of columns in matrix must equal size of vector")
        
        var y = Array(repeating: 0.0, count: v.size)
        
        switch blasImplementation {
        case .accelerate:
            fatalError("Not yet implemented")
        case .openBLAS:
            switch S.self {
            case is Double.Type:
                var A = flatten() as! [Double] // Ax = y
                var x = v.toArray(round: false) as! [Double]
                OpenBLASOperations.dgemv(Int32(n_r), Int32(n_c), &A, &x, &y)
            case is Complex.Type:
                fatalError("Not yet implemented")
            default :
                fatalError("Unsupported scalar type")
            }
        }
        
        return V(y as! [S])
    }
 
    public func toArray(round: Bool = false) -> [[S]] {
        return (0..<n_r).map { i in
                    (0..<n_c).map { j in
                        round ? self[i, j].round() : self[i, j]
                    }
                }
    }
    
    public func flatten(columnMajorOrder: Bool = false) -> [S] {
        if columnMajorOrder {
            var flattened = Array(repeating: S.zero, count: n_r * n_c)
            for i in 0..<n_r {
                for j in 0..<n_c {
                    flattened[j + n_c * i] = self[i, j]
                }
            }
            return flattened
        } else {
            return values
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

    public func approximatelyEquals(_ other: MatrixDenseBLAS<S>, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
        guard self.n_r == other.n_r && self.n_c == other.n_c else { return false }
        
        return zip(self.values, other.values).allSatisfy { $0.approximatelyEquals($1, tolerance: tolerance) }
    }
}

public enum BLASImplementation {
    case accelerate
    case openBLAS
}
