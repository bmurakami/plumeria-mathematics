public protocol TensorElementwiseArithmetic: PluTensor, TensorStructure {
    associatedtype S: PluScalar

    init(shape: [Int], initialValue: S)
    subscript(_ indices: [Int]) -> S { get set }
}

extension TensorElementwiseArithmetic where Magnitude == S.Magnitude {
    public static func + (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        var result = Self(shape: lhs.shape, initialValue: .zero)
        for index in indexCombinations(for: lhs.shape) {
            result[index] = lhs[index] + rhs[index]
        }
        return result
    }

    public static prefix func - (operand: Self) -> Self {
        var result = Self(shape: operand.shape, initialValue: .zero)
        for index in indexCombinations(for: operand.shape) {
            result[index] = -operand[index]
        }
        return result
    }

    public static func * (tensor: Self, scalar: S) -> Self {
        var result = Self(shape: tensor.shape, initialValue: .zero)
        for index in indexCombinations(for: tensor.shape) {
            result[index] = tensor[index] * scalar
        }
        return result
    }

    public static func * (scalar: S, tensor: Self) -> Self {
        tensor * scalar
    }

    public static func / (tensor: Self, scalar: S) -> Self {
        var result = Self(shape: tensor.shape, initialValue: .zero)
        for index in indexCombinations(for: tensor.shape) {
            result[index] = tensor[index] / scalar
        }
        return result
    }

    public func isApproximatelyEqual(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        guard shape == other.shape else { return false }
        for index in Self.indexCombinations(for: shape) {
            if !self[index].isApproximatelyEqual(to: other[index], relativeTolerance: relativeTolerance) {
                return false
            }
        }
        return true
    }

    private static func indexCombinations(for shape: [Int]) -> [[Int]] {
        if shape.isEmpty { return [[]] }
        if shape.contains(0) { return [] }
        return (0..<shape.reduce(1, *)).map { flatIndex in
            var remaining = flatIndex
            return shape.map { dimension in
                let index = remaining % dimension
                remaining /= dimension
                return index
            }
        }
    }
}
