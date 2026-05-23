import Numerics

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
    public func times(_ other: TensorDenseBLAS<S>, contract indices: [(left: Int, right: Int)]) -> TensorDenseBLAS<S> {
        validateContraction(indices, with: other)
        let leftContractedIndices = Set(indices.map(\.left))
        let rightContractedIndices = Set(indices.map(\.right))
        let leftFreeIndices = (0..<rank).filter { !leftContractedIndices.contains($0) }
        let rightFreeIndices = (0..<other.rank).filter { !rightContractedIndices.contains($0) }
        let leftFreeShape = leftFreeIndices.map { shape[$0] }
        let rightFreeShape = rightFreeIndices.map { other.shape[$0] }
        let contractShape = indices.map { shape[$0.left] }
        let resultShape = leftFreeShape + rightFreeShape
        if resultShape.contains(0) || contractShape.contains(0) {
            return TensorDenseBLAS(shape: resultShape, initialValue: .zero)
        }
        let leftContractIndices = indices.map(\.left)
        let rightContractIndices = indices.map(\.right)
        if leftFreeIndices + leftContractIndices == Array(0..<rank) &&
            rightContractIndices + rightFreeIndices == Array(0..<other.rank) {
            return contiguousProduct(
                resultShape: resultShape,
                leftRows: countElements(for: leftFreeShape),
                shared: countElements(for: contractShape),
                rightColumns: countElements(for: rightFreeShape),
                leftElements: elements,
                rightElements: other.elements
            )
        }
        let leftMatrix = matricizedLeftTensor(freeIndices: leftFreeIndices, contractIndices: leftContractIndices)
        let rightMatrix = other.matricizedRightTensor(
            freeIndices: rightFreeIndices,
            contractIndices: rightContractIndices
        )
        let product = matrixProduct(leftMatrix, rightMatrix)
        return TensorDenseBLAS(shape: resultShape, elements: product.flatten(columnMajorOrder: true))
    }

    public static func + (lhs: TensorDenseBLAS<S>, rhs: TensorDenseBLAS<S>) -> TensorDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return doubleTensorSum(lhs as! TensorDenseBLAS<Double>, rhs as! TensorDenseBLAS<Double>)
                as! TensorDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatTensorSum(lhs as! TensorDenseBLAS<Float>, rhs as! TensorDenseBLAS<Float>)
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            return TensorDenseBLAS<ComplexDouble>(shape: lhs.shape,
                                                  elements: complexDoubleSum(lhs.elements as! [ComplexDouble],
                                                                                  rhs.elements as! [ComplexDouble]))
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return TensorDenseBLAS<ComplexFloat>(shape: lhs.shape,
                                                 elements: complexFloatSum(lhs.elements as! [ComplexFloat],
                                                                                rhs.elements as! [ComplexFloat]))
                as! TensorDenseBLAS<S>
        }
        fatalError("Unsupported scalar type")
    }

    public static func - (lhs: TensorDenseBLAS<S>, rhs: TensorDenseBLAS<S>) -> TensorDenseBLAS<S> {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        if S.self == Double.self {
            return doubleTensorDifference(lhs as! TensorDenseBLAS<Double>, rhs as! TensorDenseBLAS<Double>)
                as! TensorDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatTensorDifference(lhs as! TensorDenseBLAS<Float>, rhs as! TensorDenseBLAS<Float>)
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
            return doubleTensorScale(tensor as! TensorDenseBLAS<Double>, by: scalar as! Double)
                as! TensorDenseBLAS<S>
        }
        if S.self == Float.self {
            return floatTensorScale(tensor as! TensorDenseBLAS<Float>, by: scalar as! Float) as! TensorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            return TensorDenseBLAS<ComplexDouble>(shape: tensor.shape,
                                                  elements: complexDoubleScale(tensor.elements as! [ComplexDouble],
                                                                                   by: scalar as! ComplexDouble))
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return TensorDenseBLAS<ComplexFloat>(shape: tensor.shape,
                                                 elements: complexFloatScale(tensor.elements as! [ComplexFloat],
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

public func multiply<S: PluScalar>(
    _ left: TensorDenseBLAS<S>, _ leftIndices: [TensorIndex],
    _ right: TensorDenseBLAS<S>, _ rightIndices: [TensorIndex]
) -> TensorDenseBLAS<S> {
    precondition(leftIndices.count == left.rank, "Left index count must match tensor rank")
    precondition(rightIndices.count == right.rank, "Right index count must match tensor rank")
    precondition(Set(leftIndices).count == leftIndices.count, "Left indices must not repeat")
    precondition(Set(rightIndices).count == rightIndices.count, "Right indices must not repeat")
    let rightPositionByIndex = Dictionary(
        uniqueKeysWithValues: rightIndices.enumerated().map { ($0.element, $0.offset) }
    )
    let contractedIndices = leftIndices.enumerated().compactMap { leftPosition, index -> (left: Int, right: Int)? in
        guard let rightPosition = rightPositionByIndex[index] else { return nil }
        return (leftPosition, rightPosition)
    }
    return left.times(right, contract: contractedIndices)
}

public func + (lhs: TensorDenseBLAS<Double>, rhs: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    doubleTensorSum(lhs, rhs)
}

public func + (lhs: TensorDenseBLAS<Float>, rhs: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    floatTensorSum(lhs, rhs)
}

public func + (lhs: TensorDenseBLAS<ComplexDouble>, rhs: TensorDenseBLAS<ComplexDouble>)
    -> TensorDenseBLAS<ComplexDouble> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<ComplexDouble>(shape: lhs.shape, elements: complexDoubleSum(lhs.elements, rhs.elements))
}

public func + (lhs: TensorDenseBLAS<ComplexFloat>, rhs: TensorDenseBLAS<ComplexFloat>)
    -> TensorDenseBLAS<ComplexFloat> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<ComplexFloat>(shape: lhs.shape, elements: complexFloatSum(lhs.elements, rhs.elements))
}

public func - (lhs: TensorDenseBLAS<Double>, rhs: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    doubleTensorDifference(lhs, rhs)
}

public func - (lhs: TensorDenseBLAS<Float>, rhs: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    floatTensorDifference(lhs, rhs)
}

public func - (lhs: TensorDenseBLAS<ComplexDouble>, rhs: TensorDenseBLAS<ComplexDouble>)
    -> TensorDenseBLAS<ComplexDouble> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<ComplexDouble>(shape: lhs.shape,
                                          elements: complexDoubleDifference(lhs.elements, rhs.elements))
}

public func - (lhs: TensorDenseBLAS<ComplexFloat>, rhs: TensorDenseBLAS<ComplexFloat>)
    -> TensorDenseBLAS<ComplexFloat> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<ComplexFloat>(shape: lhs.shape,
                                         elements: complexFloatDifference(lhs.elements, rhs.elements))
}

public prefix func - (operand: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    operand * -1.0
}

public prefix func - (operand: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    operand * -1.0
}

public prefix func - (operand: TensorDenseBLAS<ComplexDouble>) -> TensorDenseBLAS<ComplexDouble> {
    operand * -1.0
}

public prefix func - (operand: TensorDenseBLAS<ComplexFloat>) -> TensorDenseBLAS<ComplexFloat> {
    operand * -1.0
}

public func * (tensor: TensorDenseBLAS<Double>, scalar: Double) -> TensorDenseBLAS<Double> {
    doubleTensorScale(tensor, by: scalar)
}

public func * (scalar: Double, tensor: TensorDenseBLAS<Double>) -> TensorDenseBLAS<Double> {
    tensor * scalar
}

public func / (tensor: TensorDenseBLAS<Double>, scalar: Double) -> TensorDenseBLAS<Double> {
    tensor * (1 / scalar)
}

public func * (tensor: TensorDenseBLAS<Float>, scalar: Float) -> TensorDenseBLAS<Float> {
    floatTensorScale(tensor, by: scalar)
}

public func * (scalar: Float, tensor: TensorDenseBLAS<Float>) -> TensorDenseBLAS<Float> {
    tensor * scalar
}

public func / (tensor: TensorDenseBLAS<Float>, scalar: Float) -> TensorDenseBLAS<Float> {
    tensor * (1 / scalar)
}

public func * (tensor: TensorDenseBLAS<ComplexDouble>, scalar: ComplexDouble) -> TensorDenseBLAS<ComplexDouble> {
    TensorDenseBLAS<ComplexDouble>(shape: tensor.shape, elements: complexDoubleScale(tensor.elements, by: scalar))
}

public func * (scalar: ComplexDouble, tensor: TensorDenseBLAS<ComplexDouble>) -> TensorDenseBLAS<ComplexDouble> {
    tensor * scalar
}

public func / (tensor: TensorDenseBLAS<ComplexDouble>, scalar: ComplexDouble) -> TensorDenseBLAS<ComplexDouble> {
    tensor * (1 / scalar)
}

public func * (tensor: TensorDenseBLAS<ComplexFloat>, scalar: ComplexFloat) -> TensorDenseBLAS<ComplexFloat> {
    TensorDenseBLAS<ComplexFloat>(shape: tensor.shape, elements: complexFloatScale(tensor.elements, by: scalar))
}

public func * (scalar: ComplexFloat, tensor: TensorDenseBLAS<ComplexFloat>) -> TensorDenseBLAS<ComplexFloat> {
    tensor * scalar
}

public func / (tensor: TensorDenseBLAS<ComplexFloat>, scalar: ComplexFloat) -> TensorDenseBLAS<ComplexFloat> {
    tensor * (1 / scalar)
}

private func doubleTensorSum(
    _ left: TensorDenseBLAS<Double>, _ right: TensorDenseBLAS<Double>
) -> TensorDenseBLAS<Double> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Double>(shape: left.shape, elements: doubleSum(left.elements, right.elements))
}

private func floatTensorSum(
    _ left: TensorDenseBLAS<Float>, _ right: TensorDenseBLAS<Float>
) -> TensorDenseBLAS<Float> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Float>(shape: left.shape, elements: floatSum(left.elements, right.elements))
}

private func doubleTensorDifference(
    _ left: TensorDenseBLAS<Double>,
    _ right: TensorDenseBLAS<Double>
) -> TensorDenseBLAS<Double> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Double>(shape: left.shape, elements: doubleDifference(left.elements, right.elements))
}

private func floatTensorDifference(
    _ left: TensorDenseBLAS<Float>,
    _ right: TensorDenseBLAS<Float>
) -> TensorDenseBLAS<Float> {
    precondition(left.shape == right.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<Float>(shape: left.shape, elements: floatDifference(left.elements, right.elements))
}

private func doubleTensorScale(_ tensor: TensorDenseBLAS<Double>, by scalar: Double) -> TensorDenseBLAS<Double> {
    TensorDenseBLAS<Double>(shape: tensor.shape, elements: doubleScale(tensor.elements, by: scalar))
}

private func floatTensorScale(_ tensor: TensorDenseBLAS<Float>, by scalar: Float) -> TensorDenseBLAS<Float> {
    TensorDenseBLAS<Float>(shape: tensor.shape, elements: floatScale(tensor.elements, by: scalar))
}

private func matrixProduct<S: PluScalar>(
    _ left: MatrixDenseBLAS<S>, _ right: MatrixDenseBLAS<S>
) -> MatrixDenseBLAS<S> {
    if S.self == Double.self {
        return ((left as! MatrixDenseBLAS<Double>) * (right as! MatrixDenseBLAS<Double>)) as! MatrixDenseBLAS<S>
    }
    if S.self == Float.self {
        return ((left as! MatrixDenseBLAS<Float>) * (right as! MatrixDenseBLAS<Float>)) as! MatrixDenseBLAS<S>
    }
    if S.self == ComplexDouble.self {
        return ((left as! MatrixDenseBLAS<ComplexDouble>) * (right as! MatrixDenseBLAS<ComplexDouble>))
            as! MatrixDenseBLAS<S>
    }
    if S.self == ComplexFloat.self {
        return ((left as! MatrixDenseBLAS<ComplexFloat>) * (right as! MatrixDenseBLAS<ComplexFloat>))
            as! MatrixDenseBLAS<S>
    }
    fatalError("Unsupported scalar type")
}

extension TensorDenseBLAS {
    private func contiguousProduct(
        resultShape: [Int], leftRows: Int, shared: Int, rightColumns: Int, leftElements: [S], rightElements: [S]
    ) -> TensorDenseBLAS<S> {
        if S.self == Double.self {
            var result = Array(repeating: 0.0, count: leftRows * rightColumns)
            BLAS.gemm(Int32(leftRows), Int32(rightColumns), Int32(shared),
                      leftElements as! [Double], rightElements as! [Double], &result)
            return TensorDenseBLAS<Double>(shape: resultShape, elements: result) as! TensorDenseBLAS<S>
        }
        if S.self == Float.self {
            var result = Array(repeating: Float.zero, count: leftRows * rightColumns)
            BLAS.gemm(Int32(leftRows), Int32(rightColumns), Int32(shared),
                      leftElements as! [Float], rightElements as! [Float], &result)
            return TensorDenseBLAS<Float>(shape: resultShape, elements: result) as! TensorDenseBLAS<S>
        }
        if S.self == ComplexDouble.self {
            var left = BLASComplexStorage.interleaved(leftElements as! [ComplexDouble])
            var right = BLASComplexStorage.interleaved(rightElements as! [ComplexDouble])
            var result = Array(repeating: 0.0, count: leftRows * rightColumns * 2)
            BLAS.zgemm(Int32(leftRows), Int32(rightColumns), Int32(shared), &left, &right, &result)
            return TensorDenseBLAS<ComplexDouble>(shape: resultShape,
                                                  elements: BLASComplexStorage.complexValues(result))
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            var left = BLASComplexStorage.interleaved(leftElements as! [ComplexFloat])
            var right = BLASComplexStorage.interleaved(rightElements as! [ComplexFloat])
            var result = Array(repeating: Float.zero, count: leftRows * rightColumns * 2)
            BLAS.cgemm(Int32(leftRows), Int32(rightColumns), Int32(shared), &left, &right, &result)
            return TensorDenseBLAS<ComplexFloat>(shape: resultShape,
                                                 elements: BLASComplexStorage.complexValues(result))
                as! TensorDenseBLAS<S>
        }
        fatalError("Unsupported scalar type")
    }

    private func matricizedLeftTensor(freeIndices: [Int], contractIndices: [Int]) -> MatrixDenseBLAS<S> {
        let freeShape = freeIndices.map { shape[$0] }
        let contractShape = contractIndices.map { shape[$0] }
        if freeIndices + contractIndices == Array(0..<rank) {
            return MatrixDenseBLAS(rows: countElements(for: freeShape), columns: countElements(for: contractShape),
                                   values: elements)
        }
        var matrix = MatrixDenseBLAS<S>(rows: countElements(for: freeShape), columns: countElements(for: contractShape))
        for freeIndex in indexCombinations(for: freeShape) {
            for contractIndex in indexCombinations(for: contractShape) {
                var tensorIndex = Array(repeating: 0, count: rank)
                for (position, index) in freeIndices.enumerated() { tensorIndex[index] = freeIndex[position] }
                for (position, index) in contractIndices.enumerated() { tensorIndex[index] = contractIndex[position] }
                matrix[linearIndex(freeIndex, shape: freeShape), linearIndex(contractIndex, shape: contractShape)] =
                    self[tensorIndex]
            }
        }
        return matrix
    }

    private func matricizedRightTensor(freeIndices: [Int], contractIndices: [Int]) -> MatrixDenseBLAS<S> {
        let freeShape = freeIndices.map { shape[$0] }
        let contractShape = contractIndices.map { shape[$0] }
        if contractIndices + freeIndices == Array(0..<rank) {
            return MatrixDenseBLAS(rows: countElements(for: contractShape), columns: countElements(for: freeShape),
                                   values: elements)
        }
        var matrix = MatrixDenseBLAS<S>(rows: countElements(for: contractShape), columns: countElements(for: freeShape))
        for contractIndex in indexCombinations(for: contractShape) {
            for freeIndex in indexCombinations(for: freeShape) {
                var tensorIndex = Array(repeating: 0, count: rank)
                for (position, index) in contractIndices.enumerated() { tensorIndex[index] = contractIndex[position] }
                for (position, index) in freeIndices.enumerated() { tensorIndex[index] = freeIndex[position] }
                matrix[linearIndex(contractIndex, shape: contractShape), linearIndex(freeIndex, shape: freeShape)] =
                    self[tensorIndex]
            }
        }
        return matrix
    }

    private func validateContraction(_ indices: [(left: Int, right: Int)], with other: TensorDenseBLAS<S>) {
        var leftIndices = Set<Int>()
        var rightIndices = Set<Int>()
        for indexPair in indices {
            precondition(indexPair.left >= 0 && indexPair.left < rank, "Left contraction index is out of bounds")
            precondition(
                indexPair.right >= 0 && indexPair.right < other.rank,
                "Right contraction index is out of bounds"
            )
            precondition(leftIndices.insert(indexPair.left).inserted, "Left contraction indices must be unique")
            precondition(rightIndices.insert(indexPair.right).inserted, "Right contraction indices must be unique")
            precondition(shape[indexPair.left] == other.shape[indexPair.right], "Contracted dimensions must match")
        }
    }

    private func indexCombinations(for shape: [Int]) -> [[Int]] {
        if shape.isEmpty { return [[]] }
        if shape.contains(0) { return [] }
        return (0..<countElements(for: shape)).map { flatIndex in
            var remaining = flatIndex
            return shape.map { dimension in
                let index = remaining % dimension
                remaining /= dimension
                return index
            }
        }
    }

    private func linearIndex(_ indices: [Int], shape: [Int]) -> Int {
        var stride = 1
        var flatIndex = 0
        for (index, dimension) in zip(indices, shape) {
            flatIndex += index * stride
            stride *= dimension
        }
        return flatIndex
    }

    private func countElements(for shape: [Int]) -> Int { shape.reduce(1, *) }
}

private func doubleSum(_ left: [Double], _ right: [Double]) -> [Double] {
    var result = Array(repeating: 0.0, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
}

private func floatSum(_ left: [Float], _ right: [Float]) -> [Float] {
    var result = Array(repeating: Float.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
}

private func doubleDifference(_ left: [Double], _ right: [Double]) -> [Double] {
    var result = Array(repeating: 0.0, count: left.count)
    for index in 0..<left.count { result[index] = left[index] - right[index] }
    return result
}

private func floatDifference(_ left: [Float], _ right: [Float]) -> [Float] {
    var result = Array(repeating: Float.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] - right[index] }
    return result
}

private func doubleScale(_ values: [Double], by scalar: Double) -> [Double] {
    var result = Array(repeating: 0.0, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
}

private func floatScale(_ values: [Float], by scalar: Float) -> [Float] {
    var result = Array(repeating: Float.zero, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
}

private func complexDoubleSum(_ left: [ComplexDouble], _ right: [ComplexDouble]) -> [ComplexDouble] {
    var result = Array(repeating: ComplexDouble.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
}

private func complexFloatSum(_ left: [ComplexFloat], _ right: [ComplexFloat]) -> [ComplexFloat] {
    var result = Array(repeating: ComplexFloat.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
}

private func complexDoubleDifference(_ left: [ComplexDouble], _ right: [ComplexDouble]) -> [ComplexDouble] {
    var result = Array(repeating: ComplexDouble.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] - right[index] }
    return result
}

private func complexFloatDifference(_ left: [ComplexFloat], _ right: [ComplexFloat]) -> [ComplexFloat] {
    var result = Array(repeating: ComplexFloat.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] - right[index] }
    return result
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
