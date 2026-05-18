public protocol MatrixArithmetic: TensorArithmetic {
    var tr: S { get }
    var det: S { get }
    func times<V: PluVector>(_ v: V) -> V where V.S == S
    func times<M: PluMatrix>(_ m: M) -> Self where M.S == S
    func transpose() -> Self
    func inverse() -> Self
}

extension MatrixArithmetic where Self: PluMatrix {
    public var tr: S {
        precondition(rows == columns, "Trace requires a square matrix")
        var sum = S.zero
        for index in 0..<rows {
            sum += self[index, index]
        }
        return sum
    }
}

public func * <M: PluMatrix, V: PluVector>(left: M, right: V) -> V where M.S == V.S { return left.times(right) }
public func * <L: PluMatrix, R: PluMatrix>(left: L, right: R) -> L where L.S == R.S { return left.times(right) }
