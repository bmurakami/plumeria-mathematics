//public protocol Matrix {
//    associatedtype S: Scalar
//    
//    var rows: Int { get }
//    var columns: Int { get }
//    var t: Self { get }
//    subscript(i: Int, j: Int) -> S { get set }
//
//    func times<V: Vector>(_ v: V) -> V
//    func toArray() -> [[S]]
//}
//
//enum MatrixError: Error {
//    case malformedMatrix(reason: String)
//}
//
//infix operator * : MultiplicationPrecedence
//
////public func * <M: Matrix, V: Vector>(lhs: M, rhs: V) -> V where M.S == V.S {
//public func * <M: Matrix, V: Vector>(lhs: M, rhs: V) -> V {
//    return lhs.times(rhs)
//}
