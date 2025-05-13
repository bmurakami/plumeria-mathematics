public protocol Matrix {
    associatedtype Scalar
    
    var rows: Int { get }
    var columns: Int { get }
    var t: any Matrix { get }
    subscript(i: Int, j: Int) -> Scalar { get set }

    func times<V: Vector>(_ v: V) throws -> any Vector where V.Scalar == Scalar
    func toArray() -> [[Scalar]]
}

enum MatrixError: Error {
    case malformedMatrix(reason: String)
}

infix operator * : MultiplicationPrecedence

public func * <M: Matrix, V: Vector>(lhs: M, rhs: V) throws -> any Vector where M.Scalar == V.Scalar {
    return try lhs.times(rhs)
}
