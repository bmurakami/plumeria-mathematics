#if canImport(Accelerate)
import AccelerateWrapper
#endif
import Numerics
import OpenBLASWrapper

public struct VectorDenseBLAS<S: PluScalar>: TensorArithmeticBLAS {
    var view: TensorFlatView<S>
    var lazy: LazyTensor<S>?
    public var elements: [S] {
        get { lazy?.materializedElements() ?? view.contiguousElements ?? view.elements }
        set {
            view = TensorFlatView(shape: [newValue.count], elements: newValue)
            lazy = nil
        }
    }

    public init(_ values: [S]) {
        self.view = TensorFlatView(shape: [values.count], elements: values)
        self.lazy = nil
    }

    init(view: TensorFlatView<S>, lazy: LazyTensor<S>? = nil) {
        precondition(view.rank == 1, "VectorDenseBLAS requires rank 1")
        self.view = view
        self.lazy = lazy
    }

    init(size: Int, lazy: LazyTensor<S>) {
        self.view = TensorFlatView(shape: [size])
        self.lazy = lazy
    }
}

// MARK: - PluVector

extension VectorDenseBLAS: PluVector {
    public var size: Int { elements.count }

    public subscript(i: Int) -> S {
        get { lazy?.value([i]) ?? view[[i]] }
        set {
            materialize()
            view[[i]] = newValue
        }
    }

    public subscript(_ indices: [Int]) -> S {
        get {
            precondition(indices.count == 1, "Vector index rank must be 1")
            return self[indices[0]]
        }
        set {
            precondition(indices.count == 1, "Vector index rank must be 1")
            self[indices[0]] = newValue
        }
    }

    public init(_ values: TensorNestedArray<S>) {
        precondition(values.shape.count == 1, "Vector nested array must have rank 1")
        self.init(values.flatten())
    }

    public func toArray(round: Bool) -> [S] {
        if round {
            return elements.map { $0.round() }
        }
        return elements
    }

    public var shape: [Int] { [size] }
    public var rank: Int { shape.count }

    public init(shape: [Int]) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] >= 0, "Vector size must be non-negative")

        self.init(Array(repeating: .zero, count: shape[0]))
    }

    public init(shape: [Int], initialValue: S) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] >= 0, "Vector size must be non-negative")

        self.init(Array(repeating: initialValue, count: shape[0]))
    }

    public init(shape: [Int], elements: [S]) {
        precondition(shape.count == 1, "Vector shape must have rank 1")
        precondition(shape[0] == elements.count,
                     "Vector shape \(shape) requires \(shape[0]) elements, but got \(elements.count)")

        self.init(elements)
    }

    @_specialize(where S == Double)
    @_specialize(where S == Float)
    @_specialize(where S == ComplexDouble)
    @_specialize(where S == ComplexFloat)
    public func magnitude() -> S.Magnitude {
        switch S.self {
        case is Double.Type:
            return doubleMagnitude(self as! VectorDenseBLAS<Double>) as! S.Magnitude
        case is Float.Type:
            return floatMagnitude(self as! VectorDenseBLAS<Float>) as! S.Magnitude
        case is ComplexDouble.Type:
            return complexDoubleMagnitude(self as! VectorDenseBLAS<ComplexDouble>) as! S.Magnitude
        case is ComplexFloat.Type:
            return complexFloatMagnitude(self as! VectorDenseBLAS<ComplexFloat>) as! S.Magnitude
        default:
            return elements.map { $0.magnitude * $0.magnitude }.reduce(.zero, +).squareRoot()
        }
    }
}

extension VectorDenseBLAS {
    public subscript(range: Range<Int>) -> VectorDenseBLAS<S> {
        get { VectorDenseBLAS(view: materializedView().slice(SliceRange(range))) }
        set { assign(newValue, to: [TensorSliceIndex.range(range)]) }
    }

    public subscript(index: TensorSliceIndex) -> VectorDenseBLAS<S> {
        get { VectorDenseBLAS(view: materializedView().slice(index.sliceRange(dimensionSize: size))) }
        set { assign(newValue, to: [index]) }
    }

    private func materializedView() -> TensorFlatView<S> {
        if let lazy { return TensorFlatView(shape: shape, elements: lazy.materializedElements()) }
        return view
    }

    private mutating func materialize() {
        guard let lazy else { return }
        view = TensorFlatView(shape: shape, elements: lazy.materializedElements())
        self.lazy = nil
    }

    private mutating func assign(_ replacement: VectorDenseBLAS<S>, to indices: [TensorSliceIndex]) {
        materialize()
        if let lazy = replacement.lazy {
            view.assign(lazy, to: indices)
        } else {
            view.assign(replacement.view, to: indices)
        }
    }

    fileprivate var lazyTensor: LazyTensor<S> { lazy ?? .view(view) }
    fileprivate var isWholeContiguousView: Bool {
        lazy == nil && view.offset == 0 && view.isContiguous && view.storage.elements.count == size
    }

    public static func + (lhs: VectorDenseBLAS<S>, rhs: VectorDenseBLAS<S>) -> VectorDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return doubleVectorSum(lhs as! VectorDenseBLAS<Double>, rhs as! VectorDenseBLAS<Double>)
                as! VectorDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatVectorSum(lhs as! VectorDenseBLAS<Float>, rhs as! VectorDenseBLAS<Float>)
                as! VectorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            return complexDoubleVectorSum(lhs as! VectorDenseBLAS<ComplexDouble>,
                                          rhs as! VectorDenseBLAS<ComplexDouble>) as! VectorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return complexFloatVectorSum(lhs as! VectorDenseBLAS<ComplexFloat>,
                                         rhs as! VectorDenseBLAS<ComplexFloat>) as! VectorDenseBLAS<S>
        }
        fatalError("Unsupported scalar type")
    }

    public static func - (lhs: VectorDenseBLAS<S>, rhs: VectorDenseBLAS<S>) -> VectorDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return doubleVectorDifference(lhs as! VectorDenseBLAS<Double>, rhs as! VectorDenseBLAS<Double>)
                as! VectorDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatVectorDifference(lhs as! VectorDenseBLAS<Float>, rhs as! VectorDenseBLAS<Float>)
                as! VectorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            return complexDoubleVectorDifference(lhs as! VectorDenseBLAS<ComplexDouble>,
                                                 rhs as! VectorDenseBLAS<ComplexDouble>) as! VectorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return complexFloatVectorDifference(lhs as! VectorDenseBLAS<ComplexFloat>,
                                                rhs as! VectorDenseBLAS<ComplexFloat>) as! VectorDenseBLAS<S>
        }
        fatalError("Unsupported scalar type")
    }

    public static prefix func - (operand: VectorDenseBLAS<S>) -> VectorDenseBLAS<S> {
        operand * -1
    }

    public static func * (vector: VectorDenseBLAS<S>, scalar: S) -> VectorDenseBLAS<S> {
        if S.self == Double.self {
            return doubleVectorScale(vector as! VectorDenseBLAS<Double>, by: scalar as! Double)
                as! VectorDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatVectorScale(vector as! VectorDenseBLAS<Float>, by: scalar as! Float) as! VectorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            return complexDoubleVectorScale(vector as! VectorDenseBLAS<ComplexDouble>, by: scalar as! ComplexDouble)
                as! VectorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return complexFloatVectorScale(vector as! VectorDenseBLAS<ComplexFloat>, by: scalar as! ComplexFloat)
                as! VectorDenseBLAS<S>
        }
        fatalError("Unsupported scalar type")
    }

    public static func * (scalar: S, vector: VectorDenseBLAS<S>) -> VectorDenseBLAS<S> {
        vector * scalar
    }

    public static func / (vector: VectorDenseBLAS<S>, scalar: S) -> VectorDenseBLAS<S> {
        let one: S = 1
        return vector * (one / scalar)
    }

    public static func == (lhs: VectorDenseBLAS<S>, rhs: VectorDenseBLAS<S>) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements
    }
}

public func + (lhs: VectorDenseBLAS<Double>, rhs: VectorDenseBLAS<Double>) -> VectorDenseBLAS<Double> {
    doubleVectorSum(lhs, rhs)
}

public func + (lhs: VectorDenseBLAS<Float>, rhs: VectorDenseBLAS<Float>) -> VectorDenseBLAS<Float> {
    floatVectorSum(lhs, rhs)
}

public func + (lhs: VectorDenseBLAS<ComplexDouble>, rhs: VectorDenseBLAS<ComplexDouble>)
    -> VectorDenseBLAS<ComplexDouble> {
    complexDoubleVectorSum(lhs, rhs)
}

public func + (lhs: VectorDenseBLAS<ComplexFloat>, rhs: VectorDenseBLAS<ComplexFloat>)
    -> VectorDenseBLAS<ComplexFloat> {
    complexFloatVectorSum(lhs, rhs)
}

public func - (lhs: VectorDenseBLAS<Double>, rhs: VectorDenseBLAS<Double>) -> VectorDenseBLAS<Double> {
    doubleVectorDifference(lhs, rhs)
}

public func - (lhs: VectorDenseBLAS<Float>, rhs: VectorDenseBLAS<Float>) -> VectorDenseBLAS<Float> {
    floatVectorDifference(lhs, rhs)
}

public func - (lhs: VectorDenseBLAS<ComplexDouble>, rhs: VectorDenseBLAS<ComplexDouble>)
    -> VectorDenseBLAS<ComplexDouble> {
    complexDoubleVectorDifference(lhs, rhs)
}

public func - (lhs: VectorDenseBLAS<ComplexFloat>, rhs: VectorDenseBLAS<ComplexFloat>)
    -> VectorDenseBLAS<ComplexFloat> {
    complexFloatVectorDifference(lhs, rhs)
}

public prefix func - (operand: VectorDenseBLAS<Double>) -> VectorDenseBLAS<Double> {
    operand * -1.0
}

public prefix func - (operand: VectorDenseBLAS<Float>) -> VectorDenseBLAS<Float> {
    operand * -1.0
}

public prefix func - (operand: VectorDenseBLAS<ComplexDouble>) -> VectorDenseBLAS<ComplexDouble> {
    operand * -1.0
}

public prefix func - (operand: VectorDenseBLAS<ComplexFloat>) -> VectorDenseBLAS<ComplexFloat> {
    operand * -1.0
}

public func * (vector: VectorDenseBLAS<Double>, scalar: Double) -> VectorDenseBLAS<Double> {
    doubleVectorScale(vector, by: scalar)
}

public func * (scalar: Double, vector: VectorDenseBLAS<Double>) -> VectorDenseBLAS<Double> {
    vector * scalar
}

public func / (vector: VectorDenseBLAS<Double>, scalar: Double) -> VectorDenseBLAS<Double> {
    vector * (1.0 / scalar)
}

public func * (vector: VectorDenseBLAS<Float>, scalar: Float) -> VectorDenseBLAS<Float> {
    floatVectorScale(vector, by: scalar)
}

public func * (scalar: Float, vector: VectorDenseBLAS<Float>) -> VectorDenseBLAS<Float> {
    vector * scalar
}

public func / (vector: VectorDenseBLAS<Float>, scalar: Float) -> VectorDenseBLAS<Float> {
    vector * (Float(1.0) / scalar)
}

public func * (vector: VectorDenseBLAS<ComplexDouble>, scalar: ComplexDouble)
    -> VectorDenseBLAS<ComplexDouble> {
    complexDoubleVectorScale(vector, by: scalar)
}

public func * (scalar: ComplexDouble, vector: VectorDenseBLAS<ComplexDouble>)
    -> VectorDenseBLAS<ComplexDouble> {
    vector * scalar
}

public func / (vector: VectorDenseBLAS<ComplexDouble>, scalar: ComplexDouble)
    -> VectorDenseBLAS<ComplexDouble> {
    vector * (ComplexDouble(1.0, 0.0) / scalar)
}

public func * (vector: VectorDenseBLAS<ComplexFloat>, scalar: ComplexFloat) -> VectorDenseBLAS<ComplexFloat> {
    complexFloatVectorScale(vector, by: scalar)
}

public func * (scalar: ComplexFloat, vector: VectorDenseBLAS<ComplexFloat>) -> VectorDenseBLAS<ComplexFloat> {
    vector * scalar
}

public func / (vector: VectorDenseBLAS<ComplexFloat>, scalar: ComplexFloat) -> VectorDenseBLAS<ComplexFloat> {
    vector * (ComplexFloat(1.0, 0.0) / scalar)
}

private func doubleVectorSum(
    _ left: VectorDenseBLAS<Double>, _ right: VectorDenseBLAS<Double>
) -> VectorDenseBLAS<Double> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<Double>(size: left.size, lazy: left.lazyTensor.adding(right.lazyTensor))
    }
    return VectorDenseBLAS<Double>(doubleSum(left.elements, right.elements))
}

private func floatVectorSum(
    _ left: VectorDenseBLAS<Float>, _ right: VectorDenseBLAS<Float>
) -> VectorDenseBLAS<Float> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<Float>(size: left.size, lazy: left.lazyTensor.adding(right.lazyTensor))
    }
    return VectorDenseBLAS<Float>(floatSum(left.elements, right.elements))
}

private func complexDoubleVectorSum(
    _ left: VectorDenseBLAS<ComplexDouble>, _ right: VectorDenseBLAS<ComplexDouble>
) -> VectorDenseBLAS<ComplexDouble> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<ComplexDouble>(size: left.size, lazy: left.lazyTensor.adding(right.lazyTensor))
    }
    return VectorDenseBLAS<ComplexDouble>(BLASComplexStorage.sum(left.elements, right.elements))
}

private func complexFloatVectorSum(
    _ left: VectorDenseBLAS<ComplexFloat>, _ right: VectorDenseBLAS<ComplexFloat>
) -> VectorDenseBLAS<ComplexFloat> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<ComplexFloat>(size: left.size, lazy: left.lazyTensor.adding(right.lazyTensor))
    }
    return VectorDenseBLAS<ComplexFloat>(BLASComplexStorage.sum(left.elements, right.elements))
}

private func doubleVectorDifference(
    _ left: VectorDenseBLAS<Double>,
    _ right: VectorDenseBLAS<Double>
) -> VectorDenseBLAS<Double> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<Double>(size: left.size, lazy: left.lazyTensor.subtracting(right.lazyTensor))
    }
    return VectorDenseBLAS<Double>(doubleDifference(left.elements, right.elements))
}

private func floatVectorDifference(
    _ left: VectorDenseBLAS<Float>,
    _ right: VectorDenseBLAS<Float>
) -> VectorDenseBLAS<Float> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<Float>(size: left.size, lazy: left.lazyTensor.subtracting(right.lazyTensor))
    }
    return VectorDenseBLAS<Float>(floatDifference(left.elements, right.elements))
}

private func complexDoubleVectorDifference(
    _ left: VectorDenseBLAS<ComplexDouble>, _ right: VectorDenseBLAS<ComplexDouble>
) -> VectorDenseBLAS<ComplexDouble> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<ComplexDouble>(size: left.size, lazy: left.lazyTensor.subtracting(right.lazyTensor))
    }
    return VectorDenseBLAS<ComplexDouble>(BLASComplexStorage.difference(left.elements, right.elements))
}

private func complexFloatVectorDifference(
    _ left: VectorDenseBLAS<ComplexFloat>, _ right: VectorDenseBLAS<ComplexFloat>
) -> VectorDenseBLAS<ComplexFloat> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    if !left.isWholeContiguousView || !right.isWholeContiguousView {
        return VectorDenseBLAS<ComplexFloat>(size: left.size, lazy: left.lazyTensor.subtracting(right.lazyTensor))
    }
    return VectorDenseBLAS<ComplexFloat>(BLASComplexStorage.difference(left.elements, right.elements))
}

private func doubleVectorScale(_ vector: VectorDenseBLAS<Double>, by scalar: Double) -> VectorDenseBLAS<Double> {
    if !vector.isWholeContiguousView {
        return VectorDenseBLAS<Double>(size: vector.size, lazy: vector.lazyTensor.scaled(by: scalar))
    }
    return VectorDenseBLAS<Double>(doubleScale(vector.elements, by: scalar))
}

private func floatVectorScale(_ vector: VectorDenseBLAS<Float>, by scalar: Float) -> VectorDenseBLAS<Float> {
    if !vector.isWholeContiguousView {
        return VectorDenseBLAS<Float>(size: vector.size, lazy: vector.lazyTensor.scaled(by: scalar))
    }
    return VectorDenseBLAS<Float>(floatScale(vector.elements, by: scalar))
}

private func complexDoubleVectorScale(
    _ vector: VectorDenseBLAS<ComplexDouble>, by scalar: ComplexDouble
) -> VectorDenseBLAS<ComplexDouble> {
    if !vector.isWholeContiguousView {
        return VectorDenseBLAS<ComplexDouble>(size: vector.size, lazy: vector.lazyTensor.scaled(by: scalar))
    }
    return VectorDenseBLAS<ComplexDouble>(complexDoubleScale(vector.elements, by: scalar))
}

private func complexFloatVectorScale(
    _ vector: VectorDenseBLAS<ComplexFloat>, by scalar: ComplexFloat
) -> VectorDenseBLAS<ComplexFloat> {
    if !vector.isWholeContiguousView {
        return VectorDenseBLAS<ComplexFloat>(size: vector.size, lazy: vector.lazyTensor.scaled(by: scalar))
    }
    return VectorDenseBLAS<ComplexFloat>(complexFloatScale(vector.elements, by: scalar))
}

private func doubleMagnitude(_ vector: VectorDenseBLAS<Double>) -> Double {
    #if canImport(Accelerate)
    return AccelerateOperations.norm(vector.elements)
    #else
    return OpenBLASOperations.ddot(Int32(vector.size), vector.elements, vector.elements).squareRoot()
    #endif
}

private func floatMagnitude(_ vector: VectorDenseBLAS<Float>) -> Float {
    #if canImport(Accelerate)
    return AccelerateOperations.norm(vector.elements)
    #else
    return OpenBLASOperations.sdot(Int32(vector.size), vector.elements, vector.elements).squareRoot()
    #endif
}

private func complexDoubleMagnitude(_ vector: VectorDenseBLAS<ComplexDouble>) -> Double {
    #if canImport(Accelerate)
    return BLASComplexStorage.withUnsafeInterleavedStorage(vector.elements) {
        AccelerateOperations.normRaw(vector.size * 2, $0)
    }
    #else
    return BLASComplexStorage.withUnsafeInterleavedStorage(vector.elements) {
        OpenBLASOperations.ddotRaw(Int32(vector.size * 2), $0, $0).squareRoot()
    }
    #endif
}

private func complexFloatMagnitude(_ vector: VectorDenseBLAS<ComplexFloat>) -> Float {
    #if canImport(Accelerate)
    return BLASComplexStorage.withUnsafeInterleavedStorage(vector.elements) {
        AccelerateOperations.normRaw(vector.size * 2, $0)
    }
    #else
    return BLASComplexStorage.withUnsafeInterleavedStorage(vector.elements) {
        OpenBLASOperations.sdotRaw(Int32(vector.size * 2), $0, $0).squareRoot()
    }
    #endif
}

private func doubleSum(_ left: [Double], _ right: [Double]) -> [Double] {
    Array<Double>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] + right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func floatSum(_ left: [Float], _ right: [Float]) -> [Float] {
    Array<Float>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] + right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func doubleDifference(_ left: [Double], _ right: [Double]) -> [Double] {
    Array<Double>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] - right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func floatDifference(_ left: [Float], _ right: [Float]) -> [Float] {
    Array<Float>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
        left.withUnsafeBufferPointer { left in
            right.withUnsafeBufferPointer { right in
                for index in 0..<left.count { result[index] = left[index] - right[index] }
            }
        }
        initializedCount = left.count
    }
}

private func doubleScale(_ values: [Double], by scalar: Double) -> [Double] {
    Array<Double>(unsafeUninitializedCapacity: values.count) { result, initializedCount in
        values.withUnsafeBufferPointer { values in
            for index in 0..<values.count { result[index] = values[index] * scalar }
        }
        initializedCount = values.count
    }
}

private func floatScale(_ values: [Float], by scalar: Float) -> [Float] {
    Array<Float>(unsafeUninitializedCapacity: values.count) { result, initializedCount in
        values.withUnsafeBufferPointer { values in
            for index in 0..<values.count { result[index] = values[index] * scalar }
        }
        initializedCount = values.count
    }
}

private func complexDoubleScale(_ values: [ComplexDouble], by scalar: ComplexDouble) -> [ComplexDouble] {
    var result = Array(repeating: ComplexDouble.zero, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
}

private func complexFloatScale(_ values: [ComplexFloat], by scalar: ComplexFloat) -> [ComplexFloat] {
    var result = Array(repeating: ComplexFloat.zero, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
}
