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
    
    public func isApproximatelyEqual(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        guard self.size == other.size else { return false }
        
        for i in 0..<size {
            if !self[i].isApproximatelyEqual(to: other[i], relativeTolerance: relativeTolerance) {
                return false
            }
        }
        return true
    }
}
