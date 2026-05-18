private func mixedScalarIndexCombinations(for shape: [Int]) -> [[Int]] {
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

public func * <T: TensorArithmeticReference>(tensor: T, scalar: Complex) -> TensorDenseBLAS<Complex>
    where T.S == Double {
    var result = TensorDenseBLAS<Complex>(shape: tensor.shape, initialValue: .zero)
    for index in mixedScalarIndexCombinations(for: tensor.shape) {
        result[index] = tensor[index] * scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(scalar: Complex, tensor: T) -> TensorDenseBLAS<Complex>
    where T.S == Double {
    tensor * scalar
}

public func / <T: TensorArithmeticReference>(tensor: T, scalar: Complex) -> TensorDenseBLAS<Complex>
    where T.S == Double {
    var result = TensorDenseBLAS<Complex>(shape: tensor.shape, initialValue: .zero)
    for index in mixedScalarIndexCombinations(for: tensor.shape) {
        result[index] = tensor[index] / scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(tensor: T, scalar: Double) -> T where T.S == Complex {
    var result = T(shape: tensor.shape, initialValue: .zero)
    for index in mixedScalarIndexCombinations(for: tensor.shape) {
        result[index] = tensor[index] * scalar
    }
    return result
}

public func * <T: TensorArithmeticReference>(scalar: Double, tensor: T) -> T where T.S == Complex {
    tensor * scalar
}

public func / <T: TensorArithmeticReference>(tensor: T, scalar: Double) -> T where T.S == Complex {
    var result = T(shape: tensor.shape, initialValue: .zero)
    for index in mixedScalarIndexCombinations(for: tensor.shape) {
        result[index] = tensor[index] / scalar
    }
    return result
}

public func * <T: TensorArithmeticBLAS>(tensor: T, scalar: Complex) -> TensorDenseBLAS<Complex> where T.S == Double {
    TensorDenseBLAS<Complex>(shape: tensor.shape, elements: tensor.elements.map { $0 * scalar })
}

public func * <T: TensorArithmeticBLAS>(scalar: Complex, tensor: T) -> TensorDenseBLAS<Complex> where T.S == Double {
    tensor * scalar
}

public func / <T: TensorArithmeticBLAS>(tensor: T, scalar: Complex) -> TensorDenseBLAS<Complex> where T.S == Double {
    TensorDenseBLAS<Complex>(shape: tensor.shape, elements: tensor.elements.map { $0 / scalar })
}

public func * <T: TensorArithmeticBLAS>(tensor: T, scalar: Double) -> T where T.S == Complex {
    tensor * Complex(scalar, 0.0)
}

public func * <T: TensorArithmeticBLAS>(scalar: Double, tensor: T) -> T where T.S == Complex {
    tensor * scalar
}

public func / <T: TensorArithmeticBLAS>(tensor: T, scalar: Double) -> T where T.S == Complex {
    tensor / Complex(scalar, 0.0)
}
