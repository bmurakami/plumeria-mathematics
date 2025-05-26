//import COpenBLAS
//
//public struct DenseMatrix_BLAS<T: FloatingPoint & ApproximatelyEquatable>: Matrix {
//    public typealias Scalar = T
//    
//    private var n_r: Int
//    private var n_c: Int
//    private var values: [T]
//    
//    public init(_ values: [[T]]) {
//        self.n_r = values.count
//        self.n_c = values[0].count
//        self.values = [T.zero]
//        
//        self.values = (0..<n_c).flatMap { j in
//            (0..<n_r).map { i in
//                values[i][j]
//            }
//        }
//    }
//    
//    public init(rows: Int, columns: Int, initialValue: T = T.zero) {
//        self.n_r = rows
//        self.n_c = columns
//        self.values = Array(repeating: initialValue, count: rows * columns)
//    }
//
//    public var rows: Int { return n_r }
//    public var columns: Int { return n_c }
//
//    public var t: DenseMatrix_BLAS<T> {
//        var mt = DenseMatrix_BLAS(rows: n_c, columns: n_r)
//        for i in 0..<n_r {
//            for j in 0..<n_c {
//                mt[j, i] = values[i + n_r * j]
//            }
//        }
//        return mt
//    }
//
//    public func times<V>(_ v: V) -> V where V : Vector, Scalar == V.Scalar {
//        return V([T.zero])
//    }
// 
//    public subscript(i: Int, j: Int) -> T {
//        get { values[i + n_r * j] }
//        set { values[i + n_r * j] = newValue }
//    }
//
//    public func toArray() -> [[T]] {
//        return (0..<n_r).map { i in
//                    (0..<n_c).map { j in
//                        self[i, j]
//                    }
//                }
//    }
//    
//    public func flatten(columnMajorOrder: Bool = false) -> [T] {
//        if columnMajorOrder {
//            var flattened = Array(repeating: T.zero, count: n_r * n_c)
//            for i in 0..<n_r {
//                for j in 0..<n_c {
//                    flattened[j + n_c * i] = self[i, j]
//                }
//            }
//            return flattened
//        } else {
//            return values
//        }
//    }
//}
