public struct MatrixDenseReference<S: PluScalar>: PluMatrix {
    private var values: [[S]]
    
    // MARK: - PluMatrix conformance
    public var rows: Int { return values.count }
    public var columns: Int { return values[0].count }
        
    public subscript(i: Int, j: Int) -> S {
        get { return values[i][j] }
        set { values[i][j] = newValue }
    }

    public init(rows: Int, columns: Int, initialValue: S = .zero) {
        values = Array(repeating: Array(repeating: initialValue, count: columns), count: rows)
    }
    
    public init(_ values: [[S]]) {
        precondition(!values.isEmpty && !values[0].isEmpty)
        precondition(values.allSatisfy({ $0.count == values[0].count }))
        
        self.values = values
    }
        
    public func times<V: PluVector>(_ v: V) -> V where S == V.S {
        precondition(self.columns == v.size, "Matrix columns don't match vector size")

        var sum: [S] = []
        sum.reserveCapacity(self.rows)
        for i in 0..<self.rows {
            var x: S = .zero
            for j in 0..<self.columns {
                x = x + values[i][j] * v[j]
            }
            sum.append(x)
        }

        return V(sum)
    }
    
    public func transpose() -> Self {
        var mt = MatrixDenseReference(rows: self.columns, columns: self.rows, initialValue: values[0][0])
        for i in 0..<self.rows {
            for j in 0..<self.columns {
                mt[j, i] = values[i][j]
            }
        }
        return mt
    }
    
    public func toArray(round: Bool) -> [[S]] {
        if round {
            return values.map { $0.map { $0.round() }}
        }
        return values
    }
    
    public func flatten(columnMajorOrder: Bool) -> [S] {
        var flattened = Array(repeating: S.zero, count: rows * columns)
        if columnMajorOrder {
            return Array(values.joined())
        } else {
            for i in 0..<rows {
                for j in 0..<columns {
                    flattened[i + rows * j] = values[i][j]
                }
            }
            return flattened
        }
    }
    
    // MARK: - PluTensor conformance
    public static func + (lhs: MatrixDenseReference<S>, rhs: MatrixDenseReference<S>) -> MatrixDenseReference<S> {
        return MatrixDenseReference(zip(lhs.values, rhs.values).map { row1, row2 in zip(row1, row2).map(+)})
    }
    
    public static prefix func - (matrix: MatrixDenseReference<S>) -> MatrixDenseReference<S> {
        return MatrixDenseReference(matrix.values.map { $0.map { -$0 } })
    }
}
