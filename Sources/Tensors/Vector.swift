protocol Vector<S>: Tensor {
    associatedtype S: Scalar
    
    var size: Int { get }
    subscript(index: Int) -> S { get set }
    
    init(_ elements: [S])
    
//    static func zero(count: Int) -> Self
}

//public protocol Vector: Tensor { // Add Equatable
//    associatedtype S: Scalar
//    
//    init(_ values: [S])
//    var count: Int { get }
//    subscript(i: Int) -> S { get set }
//    
//    func toArray() -> [S]
//}

//extension Vector {
//    public func approximatelyEquals(_ other: Self, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
//        guard self.count == other.count else { return false }
//        
//        for i in 0..<count {
//            if !self[i].approximatelyEquals(other[i], tolerance: tolerance) {
//                return false
//            }
//        }
//        return true
//    }
//}
