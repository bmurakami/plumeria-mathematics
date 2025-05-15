public protocol Vector : Equatable {
    associatedtype Scalar : FloatingPoint
    init(_ values: [Scalar])
    var count: Int { get }
    subscript(i: Int) -> Scalar { get set }
    
    func toArray() -> [Scalar]
}

extension Vector where Scalar: ApproximatelyEquatable {
    public func approximatelyEquals(_ other: Self, tolerance: Scalar = Scalar.ulpOfOne) -> Bool {
        guard self.count == other.count else { return false }
        
        for i in 0..<count {
            if !self[i].approximatelyEquals(other[i], tolerance: tolerance) {
                return false
            }
        }
        return true
    }
}
