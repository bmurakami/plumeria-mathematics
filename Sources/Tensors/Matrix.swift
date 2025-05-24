public protocol Matrix {
    associatedtype Scalar: FloatingPoint
    
    var rows: Int { get }
    var columns: Int { get }
    var t: Self { get }
    subscript(i: Int, j: Int) -> Scalar { get set }

    func times<V: Vector>(_ v: V) -> V where V.Scalar == Scalar
    func toArray() -> [[Scalar]]
}

enum MatrixError: Error {
    case malformedMatrix(reason: String)
}

infix operator * : MultiplicationPrecedence

public func * <M: Matrix, V: Vector>(lhs: M, rhs: V) -> V where M.Scalar == V.Scalar {
    return lhs.times(rhs)
}
