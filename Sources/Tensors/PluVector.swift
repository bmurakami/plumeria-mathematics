public protocol PluVector: PluTensor, TensorStructure {
    associatedtype S: PluScalar
    
    var size: Int { get }
    subscript(index: Int) -> S { get set }
    
    init(_ elements: [S])
    
    func toArray(round: Bool) -> [S]
}

extension PluVector {
    public var shape: [Int] { [size] }
    public var rank: Int { 1 }

    public func toArray() -> [S] {
        return toArray(round: false)
    }
}
