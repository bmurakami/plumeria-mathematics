public protocol VectorType: PluTensor {
    associatedtype S: PluScalar
    
    var size: Int { get }
    subscript(index: Int) -> S { get set }
    
    init(_ elements: [S])
    
    func toArray(round: Bool) -> [S]
}

extension VectorType {
    public func approximatelyEquals(_ other: Self, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
        guard self.size == other.size else { return false }
        
        for i in 0..<size {
            if !self[i].approximatelyEquals(other[i], tolerance: tolerance) {
                return false
            }
        }
        return true
    }
}
