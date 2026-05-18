public protocol TensorArithmetic: Equatable {
    associatedtype S: PluScalar

    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static prefix func - (operand: Self) -> Self
    static func * (tensor: Self, scalar: S) -> Self
    static func * (scalar: S, tensor: Self) -> Self
    static func / (tensor: Self, scalar: S) -> Self
    func isApproximatelyEqual(to other: Self, relativeTolerance: S.Magnitude, norm: (Self) -> S.Magnitude) -> Bool
}

extension TensorArithmetic {
    public static func - (lhs: Self, rhs: Self) -> Self {
        return lhs + (-rhs)
    }
}
