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

extension PluTensor where Self: PluVector, Magnitude == S.Magnitude {
    public func isApproximatelyEqual(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        guard size == other.size else { return false }
        
        for i in 0..<size {
            if !self[i].isApproximatelyEqual(to: other[i], relativeTolerance: relativeTolerance) {
                return false
            }
        }
        return true
    }
}

extension PluTensor where Self: PluMatrix, Magnitude == S.Magnitude {
    public func isApproximatelyEqual(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        guard rows == other.rows && columns == other.columns else { return false }
        
        for i in 0..<rows {
            for j in 0..<columns {
                if !self[i, j].isApproximatelyEqual(to: other[i, j], relativeTolerance: relativeTolerance) {
                    return false
                }
            }
        }
        return true
    }
}
