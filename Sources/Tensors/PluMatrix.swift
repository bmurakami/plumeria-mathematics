public protocol PluMatrix: PluTensor {
    associatedtype S: PluScalar
    
    var rows: Int { get }
    var columns: Int { get }
    var transpose: Self { get }
    subscript(i: Int, j: Int) -> S { get set }
    
    init(rows: Int, columns: Int, initialValue: S)
    init(_ values: [[S]])

    func times<V: PluVector>(_ v: V) -> V where V.S == S
    func toArray(round: Bool) -> [[S]]
    func flatten(columnMajorOrder: Bool) -> [S]
}

infix operator * : MultiplicationPrecedence
public func * <M: PluMatrix, V: PluVector>(lhs: M, rhs: V) -> V where M.S == V.S {
    return lhs.times(rhs)
}
