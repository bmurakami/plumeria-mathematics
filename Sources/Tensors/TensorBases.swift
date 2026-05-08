public struct VectorBase<Implementation: PluVector>: PluVector {
    private var implementation: Implementation
    
    public var size: Int { implementation.size }
    
    public subscript(index: Int) -> Implementation.S {
        get { implementation[index] }
        set { implementation[index] = newValue }
    }
    
    public init(_ elements: [Implementation.S]) {
        self.implementation = Implementation(elements)
    }
    
    public init(_ implementation: Implementation) {
        self.implementation = implementation
    }
    
    public func toArray(round: Bool) -> [Implementation.S] {
        implementation.toArray(round: round)
    }
    
    public static func + (
        lhs: VectorBase<Implementation>,
        rhs: VectorBase<Implementation>
    ) -> VectorBase<Implementation> {
        VectorBase(lhs.implementation + rhs.implementation)
    }
    
    public static prefix func - (operand: VectorBase<Implementation>) -> VectorBase<Implementation> {
        VectorBase(-operand.implementation)
    }
}

public struct MatrixBase<Implementation: PluMatrix>: PluMatrix {
    private var implementation: Implementation
    
    public var rows: Int { implementation.rows }
    public var columns: Int { implementation.columns }
    
    public subscript(i: Int, j: Int) -> Implementation.S {
        get { implementation[i, j] }
        set { implementation[i, j] = newValue }
    }
    
    public init(rows: Int, columns: Int, initialValue: Implementation.S = .zero) {
        self.implementation = Implementation(rows: rows, columns: columns, initialValue: initialValue)
    }
    
    public init(_ values: [[Implementation.S]]) {
        self.implementation = Implementation(values)
    }
    
    public init(_ implementation: Implementation) {
        self.implementation = implementation
    }
    
    public func times<V: PluVector>(_ v: V) -> V where V.S == Implementation.S {
        implementation.times(v)
    }
    
    public func transpose() -> MatrixBase<Implementation> {
        MatrixBase(implementation.transpose())
    }
    
    public func toArray(round: Bool) -> [[Implementation.S]] {
        implementation.toArray(round: round)
    }
    
    public func flatten(columnMajorOrder: Bool) -> [Implementation.S] {
        implementation.flatten(columnMajorOrder: columnMajorOrder)
    }
    
    public static func + (
        lhs: MatrixBase<Implementation>,
        rhs: MatrixBase<Implementation>
    ) -> MatrixBase<Implementation> {
        MatrixBase(lhs.implementation + rhs.implementation)
    }
    
    public static prefix func - (operand: MatrixBase<Implementation>) -> MatrixBase<Implementation> {
        MatrixBase(-operand.implementation)
    }
}
