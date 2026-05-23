public struct TensorDenseBLAS<S: PluScalar>: TensorArithmeticBLAS, Equatable {
    private var view: TensorFlatView<S>

    public var shape: [Int] { view.shape }
    public var rank: Int { view.rank }
    public var elements: [S] {
        get { view.contiguousElements ?? view.elements }
        set { view = TensorFlatView(shape: shape, elements: newValue) }
    }
}

// MARK: - TensorStructure

extension TensorDenseBLAS: TensorStructure {
    public init(_ values: TensorNestedArray<S>) {
        self.init(shape: values.shape, elements: values.flatten())
    }

    public init(shape: [Int], initialValue: S = .zero) {
        self.view = TensorFlatView(shape: shape, elements: Array(repeating: initialValue, count: shape.reduce(1, *)))
    }

    public init(shape: [Int], elements: [S]) {
        self.view = TensorFlatView(shape: shape, elements: elements)
    }
}

extension TensorDenseBLAS {
    public func flatten() -> [S] { elements }

    public subscript(_ indices: [Int]) -> S {
        get { view[indices] }
        set { view[indices] = newValue }
    }

    public subscript(_ indices: Int...) -> S {
        get { self[indices] }
        set { self[indices] = newValue }
    }

    public subscript(_ indices: TensorSliceIndex...) -> TensorDenseBLAS<S> {
        get { TensorDenseBLAS(view: view.slice(indices)) }
        set { view.assign(newValue.view, to: indices) }
    }
}

// MARK: - TensorMultiplication

extension TensorDenseBLAS: TensorMultiplication {
    public typealias MatrixImplementation = MatrixDenseBLAS<S>
}

extension TensorDenseBLAS {
    public static func + (lhs: TensorDenseBLAS<S>, rhs: TensorDenseBLAS<S>) -> TensorDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return eagerDoubleTensorSum(lhs as! TensorDenseBLAS<Double>, rhs as! TensorDenseBLAS<Double>)
                as! TensorDenseBLAS<S>
        }
        if S.self == Float.self {
            return eagerFloatTensorSum(lhs as! TensorDenseBLAS<Float>, rhs as! TensorDenseBLAS<Float>)
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            return TensorDenseBLAS<ComplexDouble>(shape: lhs.shape,
                                                  elements: eagerComplexDoubleSum(lhs.elements as! [ComplexDouble],
                                                                                  rhs.elements as! [ComplexDouble]))
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return TensorDenseBLAS<ComplexFloat>(shape: lhs.shape,
                                                 elements: eagerComplexFloatSum(lhs.elements as! [ComplexFloat],
                                                                                rhs.elements as! [ComplexFloat]))
                as! TensorDenseBLAS<S>
        }
        fatalError("Unsupported scalar type")
    }

    public static func - (lhs: TensorDenseBLAS<S>, rhs: TensorDenseBLAS<S>) -> TensorDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return eagerDoubleTensorDifference(lhs as! TensorDenseBLAS<Double>, rhs as! TensorDenseBLAS<Double>)
                as! TensorDenseBLAS<S>
        }
        if S.self == Float.self {
            return eagerFloatTensorDifference(lhs as! TensorDenseBLAS<Float>, rhs as! TensorDenseBLAS<Float>)
                as! TensorDenseBLAS<S>
        }
        var result = lhs
        for index in 0..<lhs.elements.count { result.elements[index] = lhs.elements[index] - rhs.elements[index] }
        return result
    }

    public static prefix func - (operand: TensorDenseBLAS<S>) -> TensorDenseBLAS<S> {
        operand * -1
    }

    public static func * (tensor: TensorDenseBLAS<S>, scalar: S) -> TensorDenseBLAS<S> {
        if S.self == Double.self {
            return eagerDoubleTensorScale(tensor as! TensorDenseBLAS<Double>, by: scalar as! Double)
                as! TensorDenseBLAS<S>
        }
        if S.self == Float.self {
            return eagerFloatTensorScale(tensor as! TensorDenseBLAS<Float>, by: scalar as! Float) as! TensorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            return TensorDenseBLAS<ComplexDouble>(shape: tensor.shape,
                                                  elements: eagerComplexDoubleScale(tensor.elements as! [ComplexDouble],
                                                                                   by: scalar as! ComplexDouble))
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return TensorDenseBLAS<ComplexFloat>(shape: tensor.shape,
                                                 elements: eagerComplexFloatScale(tensor.elements as! [ComplexFloat],
                                                                                  by: scalar as! ComplexFloat))
                as! TensorDenseBLAS<S>
        }
        fatalError("Unsupported scalar type")
    }

    public static func * (scalar: S, tensor: TensorDenseBLAS<S>) -> TensorDenseBLAS<S> {
        tensor * scalar
    }

    public static func / (tensor: TensorDenseBLAS<S>, scalar: S) -> TensorDenseBLAS<S> {
        tensor * (1 / scalar)
    }

    public static func == (left: TensorDenseBLAS<S>, right: TensorDenseBLAS<S>) -> Bool {
        left.view == right.view
    }
}

public func + (lhs: TensorDenseBLAS<Double>, rhs: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    eagerDoubleTensorSum(lhs, rhs)
}

public func + (lhs: TensorDenseBLAS<Float>, rhs: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    eagerFloatTensorSum(lhs, rhs)
}

public func - (lhs: TensorDenseBLAS<Double>, rhs: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    eagerDoubleTensorDifference(lhs, rhs)
}

public func - (lhs: TensorDenseBLAS<Float>, rhs: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    eagerFloatTensorDifference(lhs, rhs)
}

public prefix func - (operand: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    operand * -1.0
}

public prefix func - (operand: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    operand * -1.0
}

public func * (tensor: TensorDenseBLAS<Double>, scalar: Double) -> TensorDenseBLAS<Double> {
    eagerDoubleTensorScale(tensor, by: scalar)
}

public func * (scalar: Double, tensor: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    tensor * scalar
}

public func / (tensor: TensorDenseBLAS<Double>, scalar: Double) -> TensorDenseBLAS<Double> {
    tensor * (1 / scalar)
}

public func * (tensor: TensorDenseBLAS<Float>, scalar: Float) -> TensorDenseBLAS<Float> {
    eagerFloatTensorScale(tensor, by: scalar)
}

public func * (scalar: Float, tensor: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    tensor * scalar
}

public func / (tensor: TensorDenseBLAS<Float>, scalar: Float) -> TensorDenseBLAS<Float> {
    tensor * (1 / scalar)
}

private func eagerDoubleTensorSum(
    _ left: TensorDenseBLAS<Double>, _ right: TensorDenseBLAS<Double>
) -> TensorDenseBLAS<Double> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Double>(shape: left.shape, elements: eagerDoubleSum(left.elements, right.elements))
}

private func eagerFloatTensorSum(
    _ left: TensorDenseBLAS<Float>, _ right: TensorDenseBLAS<Float>
) -> TensorDenseBLAS<Float> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Float>(shape: left.shape, elements: eagerFloatSum(left.elements, right.elements))
}

private func eagerDoubleTensorDifference(
    _ left: TensorDenseBLAS<Double>,
    _ right: TensorDenseBLAS<Double>
) -> TensorDenseBLAS<Double> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Double>(shape: left.shape, elements: eagerDoubleDifference(left.elements, right.elements))
}

private func eagerFloatTensorDifference(
    _ left: TensorDenseBLAS<Float>,
    _ right: TensorDenseBLAS<Float>
) -> TensorDenseBLAS<Float> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Float>(shape: left.shape, elements: eagerFloatDifference(left.elements, right.elements))
}

private func eagerDoubleTensorScale(_ tensor: TensorDenseBLAS<Double>, by scalar: Double) -> TensorDenseBLAS<Double> {
    TensorDenseBLAS<Double>(shape: tensor.shape, elements: eagerDoubleScale(tensor.elements, by: scalar))
}

private func eagerFloatTensorScale(_ tensor: TensorDenseBLAS<Float>, by scalar: Float) -> TensorDenseBLAS<Float> {
    TensorDenseBLAS<Float>(shape: tensor.shape, elements: eagerFloatScale(tensor.elements, by: scalar))
}

private func eagerDoubleSum(_ left: [Double], _ right: [Double]) -> [Double] {
    var result = Array(repeating: 0.0, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
}

private func eagerFloatSum(_ left: [Float], _ right: [Float]) -> [Float] {
    var result = Array(repeating: Float.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
}

private func eagerDoubleDifference(_ left: [Double], _ right: [Double]) -> [Double] {
    var result = Array(repeating: 0.0, count: left.count)
    for index in 0..<left.count { result[index] = left[index] - right[index] }
    return result
}

private func eagerFloatDifference(_ left: [Float], _ right: [Float]) -> [Float] {
    var result = Array(repeating: Float.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] - right[index] }
    return result
}

private func eagerDoubleScale(_ values: [Double], by scalar: Double) -> [Double] {
    var result = Array(repeating: 0.0, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
}

private func eagerFloatScale(_ values: [Float], by scalar: Float) -> [Float] {
    var result = Array(repeating: Float.zero, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
}

private func eagerComplexDoubleSum(_ left: [ComplexDouble], _ right: [ComplexDouble]) -> [ComplexDouble] {
    var x = BLASComplexStorage.interleaved(right)
    var y = BLASComplexStorage.interleaved(left)
    BLAS.zaxpy(Int32(right.count), &x, &y)
    return BLASComplexStorage.complexValues(y)
}

private func eagerComplexFloatSum(_ left: [ComplexFloat], _ right: [ComplexFloat]) -> [ComplexFloat] {
    var x = BLASComplexStorage.interleaved(right)
    var y = BLASComplexStorage.interleaved(left)
    BLAS.caxpy(Int32(right.count), &x, &y)
    return BLASComplexStorage.complexValues(y)
}

private func eagerComplexDoubleScale(_ values: [ComplexDouble], by scalar: ComplexDouble) -> [ComplexDouble] {
    var result = BLASComplexStorage.interleaved(values)
    var alpha = BLASComplexStorage.interleaved([scalar])
    BLAS.zscal(Int32(values.count), &alpha, &result)
    return BLASComplexStorage.complexValues(result)
}

private func eagerComplexFloatScale(_ values: [ComplexFloat], by scalar: ComplexFloat) -> [ComplexFloat] {
    var result = BLASComplexStorage.interleaved(values)
    var alpha = BLASComplexStorage.interleaved([scalar])
    BLAS.cscal(Int32(values.count), &alpha, &result)
    return BLASComplexStorage.complexValues(result)
}
