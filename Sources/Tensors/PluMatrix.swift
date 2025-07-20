public protocol PluMatrix: PluTensor {
    associatedtype S: PluScalar
    
    var rows: Int { get }
    var columns: Int { get }
    subscript(i: Int, j: Int) -> S { get set }
    
    init(rows: Int, columns: Int, initialValue: S)
    init(_ values: [[S]])

    func times<V: PluVector>(_ v: V) -> V where V.S == S
    func transpose() -> Self
    func toArray(round: Bool) -> [[S]]
    func flatten(columnMajorOrder: Bool) -> [S]
}

extension PluMatrix {
    public func toArray() -> [[S]] {
        return toArray(round: false)
    }
    
    public func flatten() -> [S] {
        return flatten(columnMajorOrder: false)
    }
    
    // Temporary
    public func isApproximatelyEqual(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        return true
    }
}

infix operator * : MultiplicationPrecedence
public func * <M: PluMatrix, V: PluVector>(lhs: M, rhs: V) -> V where M.S == V.S {
    return lhs.times(rhs)
}
