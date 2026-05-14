public protocol PluMatrix: PluTensor, TensorStructure {
    associatedtype S: PluScalar
    
    var rows: Int { get }
    var columns: Int { get }
    subscript(i: Int, j: Int) -> S { get set }
    
    init(rows: Int, columns: Int, initialValue: S)
    init(_ values: [[S]])

    func times<V: PluVector>(_ v: V) -> V where V.S == S
    func times<M: PluMatrix>(_ m: M) -> Self where M.S == S
    func transpose() -> Self
    func toArray(round: Bool) -> [[S]]
    func flatten(columnMajorOrder: Bool) -> [S]
}

extension PluMatrix {
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { 2 }

    public func toArray() -> [[S]] { return toArray(round: false) }
    public func flatten() -> [S] { return flatten(columnMajorOrder: true) }
}

infix operator * : MultiplicationPrecedence

public func * <M: PluMatrix, V: PluVector>(lhs: M, rhs: V) -> V where M.S == V.S { return lhs.times(rhs) }
public func * <L: PluMatrix, R: PluMatrix>(lhs: L, rhs: R) -> L where L.S == R.S { return lhs.times(rhs) }
