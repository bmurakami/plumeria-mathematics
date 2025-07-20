public protocol PluTensor: Equatable {
    associatedtype Magnitude: FloatingPoint
    
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static prefix func - (operand: Self) -> Self
    func isApproximatelyEqual(to other: Self, relativeTolerance: Magnitude,  norm: (Self) -> Magnitude) -> Bool
}

extension PluTensor {
    public static func - (lhs: Self, rhs: Self) -> Self {
        return lhs + (-rhs)
    }
}
