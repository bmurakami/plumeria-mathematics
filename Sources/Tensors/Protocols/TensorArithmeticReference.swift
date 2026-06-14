public protocol TensorArithmeticReference: TensorArithmetic, TensorStructure where S: PluScalar {
    init(shape: [Int], initialValue: S)
    subscript(_ indices: [Int]) -> S { get set }
}

extension TensorArithmeticReference {
    public static func + (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        var result = Self(shape: lhs.shape, initialValue: .zero)
        for i in indexCombinations(for: lhs.shape) {
            result[i] = lhs[i] + rhs[i]
        }
        return result
    }

    public static prefix func - (operand: Self) -> Self {
        var result = Self(shape: operand.shape, initialValue: .zero)
        for i in indexCombinations(for: operand.shape) {
            result[i] = -operand[i]
        }
        return result
    }

    public static func * (tensor: Self, scalar: S) -> Self {
        var result = Self(shape: tensor.shape, initialValue: .zero)
        for i in indexCombinations(for: tensor.shape) {
            result[i] = tensor[i] * scalar
        }
        return result
    }

    public static func * (scalar: S, tensor: Self) -> Self {
        tensor * scalar
    }

    public static func / (tensor: Self, scalar: S) -> Self {
        var result = Self(shape: tensor.shape, initialValue: .zero)
        for i in indexCombinations(for: tensor.shape) {
            result[i] = tensor[i] / scalar
        }
        return result
    }

    public func isClose(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        guard shape == other.shape else { return false }
        for i in Self.indexCombinations(for: shape) {
            if !self[i].isClose(to: other[i], relativeTolerance: relativeTolerance) {
                return false
            }
        }
        return true
    }

    private static func indexCombinations(for shape: [Int]) -> [[Int]] {
        // Example: indexCombinations(for: [2, 3]) returns [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]].
        if shape.isEmpty { return [[]] }
        if shape.contains(0) { return [] }
        return (0..<shape.reduce(1, *)).map { flatIndex in
            var remaining = flatIndex
            return shape.map { dimension in
                let i = remaining % dimension
                remaining /= dimension
                return i
            }
        }
    }
}
