public protocol Matrix {
    associatedtype Scalar
    
    var rows: Int { get }
    var columns: Int { get }
    var t: any Matrix { get }

    subscript(i: Int, j: Int) -> Scalar { get set }
}

enum MatrixError: Error {
    case malformedMatrix(reason: String)
}
