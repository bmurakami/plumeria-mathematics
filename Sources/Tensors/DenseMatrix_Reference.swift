//public struct DenseMatrix_Reference<S: Scalar>: Matrix {
//    public private(set)var values: [[S]]
//
//    public init(rows: Int, columns: Int, intialValue: S = 0.0) {
//        values = Array(repeating: Array(repeating: intialValue, count: columns), count: rows)
//    }
//
//    public init(_ values: [[S]]) {
//        precondition(!values.isEmpty && !values[0].isEmpty,
//                     "The matrix cannot be empty or partially empty")
//        precondition(values.allSatisfy { $0.count == values[0].count},
//                     "All rows in a matrix must have the same size")
//        
//        self.values = values
//    }
//    
//    public func flatten(columnMajorOrder: Bool = false) -> [S] {
//        var flattened = Array(repeating: S.zero, count: rows * columns)
//        if columnMajorOrder {
//            return Array(values.joined())
//        } else {
//            for i in 0..<rows {
//                for j in 0..<columns {
//                    flattened[i + rows * j] = values[i][j]
//                }
//            }
//            return flattened
//        }
//    }
//    
//    public var rows: Int { return values.count }
//    public var columns: Int { return values[0].count }
//    
//    public func times<V: Vector>(_ v: V) -> V {
//        precondition(self.columns == v.count, "Matrix columns don't match vector size")
//        
//        var sum: [S] = []
//        sum.reserveCapacity(self.rows)
//        for i in 0..<self.rows {
//            var x: T = .zero
//            for j in 0..<self.columns {
//                x += values[i][j] * v[j]
//            }
//            sum.append(x)
//        }
//        
//        return V(sum)
//    }
//
//    public var t: Self {
//        var At = DenseMatrix_Reference<T>(rows: self.columns, columns: self.rows, intialValue: values[0][0])
//        for i in 0..<self.rows {
//            for j in 0..<self.columns {
//                At[j, i] = values[i][j]
//            }
//        }
//        return At
//    }
//    
//    public subscript(i: Int, j: Int) -> Scalar {
//        get { return values[i][j] }
//        set { values[i][j] = newValue }
//    }
//    
//    public func toArray() -> [[Scalar]] {
//        return values
//    }
//}
