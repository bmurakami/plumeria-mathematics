public protocol Tensor: Equatable {
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static prefix func - (operand: Self) -> Self
    func approximatelyEquals(_ other: Self, tolerance: Double) -> Bool
}

public extension Tensor {
    static func - (lhs: Self, rhs: Self) -> Self {
        return lhs + (-rhs)
    }

}
