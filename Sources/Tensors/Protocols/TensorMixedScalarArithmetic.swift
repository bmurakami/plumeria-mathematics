import Numerics

private func mixedScalarIndexCombinations(for shape: [Int]) -> [[Int]] {
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

public func * <T: TensorArithmeticReference>(tensor: T, scalar: ComplexDouble) -> TensorDenseBLAS<ComplexDouble>
    where T.S == Double {
    var result = TensorDenseBLAS<ComplexDouble>(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] * scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(scalar: ComplexDouble, tensor: T) -> TensorDenseBLAS<ComplexDouble>
    where T.S == Double {
    tensor * scalar
}

public func / <T: TensorArithmeticReference>(tensor: T, scalar: ComplexDouble) -> TensorDenseBLAS<ComplexDouble>
    where T.S == Double {
    var result = TensorDenseBLAS<ComplexDouble>(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] / scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(tensor: T, scalar: Double) -> T where T.S == ComplexDouble {
    var result = T(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] * scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(scalar: Double, tensor: T) -> T where T.S == ComplexDouble {
    tensor * scalar
}

public func / <T: TensorArithmeticReference>(tensor: T, scalar: Double) -> T where T.S == ComplexDouble {
    var result = T(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] / scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(tensor: T, scalar: ComplexFloat) -> TensorDenseBLAS<ComplexFloat>
    where T.S == Float {
    var result = TensorDenseBLAS<ComplexFloat>(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] * scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(scalar: ComplexFloat, tensor: T) -> TensorDenseBLAS<ComplexFloat>
    where T.S == Float {
    tensor * scalar
}

public func / <T: TensorArithmeticReference>(tensor: T, scalar: ComplexFloat) -> TensorDenseBLAS<ComplexFloat>
    where T.S == Float {
    var result = TensorDenseBLAS<ComplexFloat>(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] / scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(tensor: T, scalar: Float) -> T where T.S == ComplexFloat {
    var result = T(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] * scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(scalar: Float, tensor: T) -> T where T.S == ComplexFloat {
    tensor * scalar
}

public func / <T: TensorArithmeticReference>(tensor: T, scalar: Float) -> T where T.S == ComplexFloat {
    var result = T(shape: tensor.shape, initialValue: .zero)
    for i in mixedScalarIndexCombinations(for: tensor.shape) {
        result[i] = tensor[i] / scalar
    }
    return result
}

public func * <T: TensorArithmeticBLAS>(tensor: T, scalar: ComplexDouble) -> TensorDenseBLAS<ComplexDouble>
    where T.S == Double {
    TensorDenseBLAS<ComplexDouble>(shape: tensor.shape, elements: tensor.elements.map { $0 * scalar })
}

public func * <T: TensorArithmeticBLAS>(scalar: ComplexDouble, tensor: T) -> TensorDenseBLAS<ComplexDouble>
    where T.S == Double {
    tensor * scalar
}

public func / <T: TensorArithmeticBLAS>(tensor: T, scalar: ComplexDouble) -> TensorDenseBLAS<ComplexDouble>
    where T.S == Double {
    TensorDenseBLAS<ComplexDouble>(shape: tensor.shape, elements: tensor.elements.map { $0 / scalar })
}

public func * <T: TensorArithmeticBLAS>(tensor: T, scalar: Double) -> T where T.S == ComplexDouble {
    tensor * ComplexDouble(scalar, 0.0)
}

public func * <T: TensorArithmeticBLAS>(scalar: Double, tensor: T) -> T where T.S == ComplexDouble {
    tensor * scalar
}

public func / <T: TensorArithmeticBLAS>(tensor: T, scalar: Double) -> T where T.S == ComplexDouble {
    tensor / ComplexDouble(scalar, 0.0)
}

public func * <T: TensorArithmeticBLAS>(tensor: T, scalar: ComplexFloat) -> TensorDenseBLAS<ComplexFloat>
    where T.S == Float {
    TensorDenseBLAS<ComplexFloat>(shape: tensor.shape, elements: tensor.elements.map { $0 * scalar })
}

public func * <T: TensorArithmeticBLAS>(scalar: ComplexFloat, tensor: T) -> TensorDenseBLAS<ComplexFloat>
    where T.S == Float {
    tensor * scalar
}

public func / <T: TensorArithmeticBLAS>(tensor: T, scalar: ComplexFloat) -> TensorDenseBLAS<ComplexFloat>
    where T.S == Float {
    TensorDenseBLAS<ComplexFloat>(shape: tensor.shape, elements: tensor.elements.map { $0 / scalar })
}

public func * <T: TensorArithmeticBLAS>(tensor: T, scalar: Float) -> T where T.S == ComplexFloat {
    tensor * ComplexFloat(scalar, 0.0)
}

public func * <T: TensorArithmeticBLAS>(scalar: Float, tensor: T) -> T where T.S == ComplexFloat {
    tensor * scalar
}

public func / <T: TensorArithmeticBLAS>(tensor: T, scalar: Float) -> T where T.S == ComplexFloat {
    tensor / ComplexFloat(scalar, 0.0)
}
