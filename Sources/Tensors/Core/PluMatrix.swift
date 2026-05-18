public protocol PluMatrix: PluTensor, TensorStructure, MatrixArithmetic where S: PluScalar {
    var rows: Int { get }
    var columns: Int { get }
    subscript(i: Int, j: Int) -> S { get set }

    init(rows: Int, columns: Int, initialValue: S)
    init(_ elements: [[S]])

    func toArray(round: Bool) -> [[S]]
    func flatten(columnMajorOrder: Bool) -> [S]
}

extension PluMatrix {
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { 2 }
    public var t: Self { transpose() }
}

extension PluMatrix {
    public func toArray() -> [[S]] { return toArray(round: false) }
    public func flatten() -> [S] { return flatten(columnMajorOrder: true) }
    public static func identity(size: Int) -> Self {
        precondition(size > 0, "Identity matrix size must be positive")
        return Self((0..<size).map { row in
            (0..<size).map { column in row == column ? 1 : 0 }
        })
    }
}
