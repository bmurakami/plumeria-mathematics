public protocol Vector : Equatable {
    associatedtype Scalar : FloatingPoint
    
    var count: Int { get }
    subscript(i: Int) -> Scalar { get set }
}

extension Vector where Scalar: ApproximatelyEquatable {
    public func approximatelyEquals(_ other: Self) -> Bool {
        guard self.count == other.count else { return false }
        
        for i in 0..<count {
            if !self[i].approximatelyEquals(other[i]) {
                return false
            }
        }
        return true
    }
}

