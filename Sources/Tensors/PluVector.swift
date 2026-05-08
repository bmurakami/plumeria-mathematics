public protocol PluVector: PluTensor {
    associatedtype S: PluScalar
    
    var size: Int { get }
    subscript(index: Int) -> S { get set }
    
    init(_ elements: [S])
    
    func toArray(round: Bool) -> [S]
}

extension PluVector {
    public func toArray() -> [S] {
        return toArray(round: false)
    }
}
