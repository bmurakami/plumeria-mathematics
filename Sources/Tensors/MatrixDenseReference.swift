public struct MatrixDenseReference<S: PluScalar>: PluMatrix, TensorElementwiseArithmetic {
    private var values: [[S]]
    
    // MARK: - PluMatrix conformance
    public var rows: Int { return values.count }
    public var columns: Int { return values[0].count }
        
    public subscript(i: Int, j: Int) -> S {
        get { return values[i][j] }
        set { values[i][j] = newValue }
    }

    public subscript(_ indices: [Int]) -> S {
        get {
            precondition(indices.count == 2, "Matrix index rank must be 2")
            return self[indices[0], indices[1]]
        }
        set {
            precondition(indices.count == 2, "Matrix index rank must be 2")
            self[indices[0], indices[1]] = newValue
        }
    }

    public init(rows: Int, columns: Int, initialValue: S = .zero) {
        values = Array(repeating: Array(repeating: initialValue, count: columns), count: rows)
    }

    public init(shape: [Int], initialValue: S) {
        precondition(shape.count == 2, "MatrixDenseReference shape must have rank 2")
        precondition(shape.allSatisfy { $0 >= 0 }, "Matrix shape dimensions must be non-negative")

        self.init(rows: shape[0], columns: shape[1], initialValue: initialValue)
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

    public func times<M: PluMatrix>(_ m: M) -> MatrixDenseReference<S> where M.S == S {
        precondition(columns == m.rows, "Matrix columns must match matrix rows")
        var product = MatrixDenseReference(rows: rows, columns: m.columns, initialValue: .zero)
        for i in 0..<rows {
            for j in 0..<m.columns {
                var sum = S.zero
                for k in 0..<columns {
                    sum += values[i][k] * m[k, j]
                }
                product[i, j] = sum
            }
        }
        return product
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
            for i in 0..<rows {
                for j in 0..<columns {
                    flattened[i + rows * j] = values[i][j]
                }
            }
            return flattened
        } else {
            return Array(values.joined())
        }
    }
    
}
