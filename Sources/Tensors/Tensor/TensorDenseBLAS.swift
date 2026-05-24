#if canImport(Accelerate)
import AccelerateWrapper
#endif
import Numerics

private struct TensorDenseContractionPlan {
    let leftFreeIndices: [Int]
    let rightFreeIndices: [Int]
    let leftContractIndices: [Int]
    let rightContractIndices: [Int]
    let leftFreeShape: [Int]
    let rightFreeShape: [Int]
    let contractShape: [Int]
    var resultShape: [Int] { leftFreeShape + rightFreeShape }
}

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
    @specialized(where S == Double)
    @specialized(where S == Float)
    @specialized(where S == ComplexDouble)
    @specialized(where S == ComplexFloat)
    public func times(_ other: TensorDenseBLAS<S>, contract indices: [(left: Int, right: Int)]) -> TensorDenseBLAS<S> {
        let plan = contractionPlan(with: other, contract: indices)
        if plan.resultShape.contains(0) || plan.contractShape.contains(0) {
            return TensorDenseBLAS(shape: plan.resultShape, initialValue: .zero)
        }
        if areStorageOrdered(plan.leftFreeIndices, plan.leftContractIndices, rank: rank) &&
            areStorageOrdered(plan.rightContractIndices, plan.rightFreeIndices, rank: other.rank) {
            let leftMatrix = MatrixDenseBLAS<S>(rows: countElements(for: plan.leftFreeShape),
                                                columns: countElements(for: plan.contractShape), values: elements)
            let rightMatrix = MatrixDenseBLAS<S>(rows: countElements(for: plan.contractShape),
                                                 columns: countElements(for: plan.rightFreeShape),
                                                 values: other.elements)
            let product = matrixProduct(leftMatrix, rightMatrix)
            return TensorDenseBLAS(shape: plan.resultShape, elements: product.flatten(columnMajorOrder: true))
        }
        let leftMatrix = matricizedLeftTensor(freeIndices: plan.leftFreeIndices,
                                              contractIndices: plan.leftContractIndices)
        let rightMatrix = other.matricizedRightTensor(freeIndices: plan.rightFreeIndices,
                                                      contractIndices: plan.rightContractIndices)
        let product = matrixProduct(leftMatrix, rightMatrix)
        return TensorDenseBLAS(shape: plan.resultShape, elements: product.flatten(columnMajorOrder: true))
    }

    fileprivate func contractionPlan(
        with other: TensorDenseBLAS<S>, contract indices: [(left: Int, right: Int)]
    ) -> TensorDenseContractionPlan {
        var leftContracted = Array(repeating: false, count: rank)
        var rightContracted = Array(repeating: false, count: other.rank)
        var leftContractIndices: [Int] = []
        var rightContractIndices: [Int] = []
        var contractShape: [Int] = []
        leftContractIndices.reserveCapacity(indices.count)
        rightContractIndices.reserveCapacity(indices.count)
        contractShape.reserveCapacity(indices.count)
        for indexPair in indices {
            precondition(indexPair.left >= 0 && indexPair.left < rank, "Left contraction index is out of bounds")
            precondition(
                indexPair.right >= 0 && indexPair.right < other.rank,
                "Right contraction index is out of bounds"
            )
            precondition(!leftContracted[indexPair.left], "Left contraction indices must be unique")
            precondition(!rightContracted[indexPair.right], "Right contraction indices must be unique")
            precondition(shape[indexPair.left] == other.shape[indexPair.right], "Contracted dimensions must match")
            leftContracted[indexPair.left] = true
            rightContracted[indexPair.right] = true
            leftContractIndices.append(indexPair.left)
            rightContractIndices.append(indexPair.right)
            contractShape.append(shape[indexPair.left])
        }
        var leftFreeIndices: [Int] = []
        var rightFreeIndices: [Int] = []
        var leftFreeShape: [Int] = []
        var rightFreeShape: [Int] = []
        for index in 0..<rank where !leftContracted[index] {
            leftFreeIndices.append(index)
            leftFreeShape.append(shape[index])
        }
        for index in 0..<other.rank where !rightContracted[index] {
            rightFreeIndices.append(index)
            rightFreeShape.append(other.shape[index])
        }
        return TensorDenseContractionPlan(
            leftFreeIndices: leftFreeIndices,
            rightFreeIndices: rightFreeIndices,
            leftContractIndices: leftContractIndices,
            rightContractIndices: rightContractIndices,
            leftFreeShape: leftFreeShape,
            rightFreeShape: rightFreeShape,
            contractShape: contractShape
        )
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
                                                  elements: BLASComplexStorage.sum(lhs.elements as! [ComplexDouble],
                                                                                  rhs.elements as! [ComplexDouble]))
                as! TensorDenseBLAS<S>
        }
        if S.self == ComplexFloat.self {
            return TensorDenseBLAS<ComplexFloat>(shape: lhs.shape,
                                                 elements: BLASComplexStorage.sum(lhs.elements as! [ComplexFloat],
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

public func multiply(
    _ left: TensorDenseBLAS<Double>, _ leftIndices: [TensorIndex],
    _ right: TensorDenseBLAS<Double>, _ rightIndices: [TensorIndex]
) -> TensorDenseBLAS<Double> {
    tensorDenseBLASMultiply(left, leftIndices, right, rightIndices, product: *)
}

public func multiply(
    _ left: TensorDenseBLAS<Float>, _ leftIndices: [TensorIndex],
    _ right: TensorDenseBLAS<Float>, _ rightIndices: [TensorIndex]
) -> TensorDenseBLAS<Float> {
    tensorDenseBLASMultiply(left, leftIndices, right, rightIndices, product: *)
}

public func multiply(
    _ left: TensorDenseBLAS<ComplexDouble>, _ leftIndices: [TensorIndex],
    _ right: TensorDenseBLAS<ComplexDouble>, _ rightIndices: [TensorIndex]
) -> TensorDenseBLAS<ComplexDouble> {
    tensorDenseBLASMultiply(left, leftIndices, right, rightIndices, product: *)
}

public func multiply(
    _ left: TensorDenseBLAS<ComplexFloat>, _ leftIndices: [TensorIndex],
    _ right: TensorDenseBLAS<ComplexFloat>, _ rightIndices: [TensorIndex]
) -> TensorDenseBLAS<ComplexFloat> {
    tensorDenseBLASMultiply(left, leftIndices, right, rightIndices, product: *)
}

public func multiply(_ left: TensorDenseBLAS<Double>, _ right: TensorDenseBLAS<Double>, _ notation: String)
    -> TensorDenseBLAS<Double> {
    let (leftIndices, rightIndices) = denseTensorIndices(notation)
    return multiply(left, leftIndices, right, rightIndices)
}

public func multiply(_ left: TensorDenseBLAS<Float>, _ right: TensorDenseBLAS<Float>, _ notation: String)
    -> TensorDenseBLAS<Float> {
    let (leftIndices, rightIndices) = denseTensorIndices(notation)
    return multiply(left, leftIndices, right, rightIndices)
}

public func multiply(
    _ left: TensorDenseBLAS<ComplexDouble>, _ right: TensorDenseBLAS<ComplexDouble>, _ notation: String
) -> TensorDenseBLAS<ComplexDouble> {
    let (leftIndices, rightIndices) = denseTensorIndices(notation)
    return multiply(left, leftIndices, right, rightIndices)
}

public func multiply(
    _ left: TensorDenseBLAS<ComplexFloat>, _ right: TensorDenseBLAS<ComplexFloat>, _ notation: String
) -> TensorDenseBLAS<ComplexFloat> {
    let (leftIndices, rightIndices) = denseTensorIndices(notation)
    return multiply(left, leftIndices, right, rightIndices)
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
    return TensorDenseBLAS<ComplexDouble>(shape: lhs.shape,
                                          elements: BLASComplexStorage.sum(lhs.elements, rhs.elements))
}

public func + (lhs: TensorDenseBLAS<ComplexFloat>, rhs: TensorDenseBLAS<ComplexFloat>)
    -> TensorDenseBLAS<ComplexFloat> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<ComplexFloat>(shape: lhs.shape, elements: BLASComplexStorage.sum(lhs.elements, rhs.elements))
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
                                          elements: BLASComplexStorage.difference(lhs.elements, rhs.elements))
}

public func - (lhs: TensorDenseBLAS<ComplexFloat>, rhs: TensorDenseBLAS<ComplexFloat>)
    -> TensorDenseBLAS<ComplexFloat> {
    precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
    return TensorDenseBLAS<ComplexFloat>(shape: lhs.shape,
                                         elements: BLASComplexStorage.difference(lhs.elements, rhs.elements))
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

private func tensorDenseBLASMultiply<S: PluScalar>(
    _ left: TensorDenseBLAS<S>, _ leftIndices: [TensorIndex],
    _ right: TensorDenseBLAS<S>, _ rightIndices: [TensorIndex],
    product: (MatrixDenseBLAS<S>, MatrixDenseBLAS<S>) -> MatrixDenseBLAS<S>
) -> TensorDenseBLAS<S> {
    let contractIndices = denseTensorContractIndices(left, leftIndices, right, rightIndices)
    let plan = left.contractionPlan(with: right, contract: contractIndices)
    if plan.resultShape.contains(0) || plan.contractShape.contains(0) {
        return TensorDenseBLAS(shape: plan.resultShape, initialValue: .zero)
    }
    let leftMatrix = left.matricizedLeftTensor(freeIndices: plan.leftFreeIndices,
                                               contractIndices: plan.leftContractIndices)
    let rightMatrix = right.matricizedRightTensor(freeIndices: plan.rightFreeIndices,
                                                  contractIndices: plan.rightContractIndices)
    let matrix = product(leftMatrix, rightMatrix)
    return TensorDenseBLAS(shape: plan.resultShape, elements: matrix.flatten(columnMajorOrder: true))
}

private func denseTensorContractIndices<S: PluScalar>(
    _ left: TensorDenseBLAS<S>, _ leftIndices: [TensorIndex],
    _ right: TensorDenseBLAS<S>, _ rightIndices: [TensorIndex]
) -> [(left: Int, right: Int)] {
    precondition(leftIndices.count == left.rank, "Left index count must match tensor rank")
    precondition(rightIndices.count == right.rank, "Right index count must match tensor rank")
    precondition(Set(leftIndices).count == leftIndices.count, "Left indices must not repeat")
    precondition(Set(rightIndices).count == rightIndices.count, "Right indices must not repeat")
    let rightPositionByIndex = Dictionary(
        uniqueKeysWithValues: rightIndices.enumerated().map { ($0.element, $0.offset) }
    )
    return leftIndices.enumerated().compactMap { leftPosition, index -> (left: Int, right: Int)? in
        guard let rightPosition = rightPositionByIndex[index] else { return nil }
        return (leftPosition, rightPosition)
    }
}

private func denseTensorIndices(_ notation: String) -> ([TensorIndex], [TensorIndex]) {
    let compact = notation.filter { !$0.isWhitespace }
    precondition(!compact.contains("->"), "Tensor multiplication notation must not include an output clause")
    let operands = compact.split(separator: ",", omittingEmptySubsequences: false)
    precondition(operands.count == 2, "Tensor multiplication notation must contain two operands")
    return (operands[0].map { TensorIndex(String($0)) }, operands[1].map { TensorIndex(String($0)) })
}

extension TensorDenseBLAS {
    fileprivate func matricizedLeftTensor(freeIndices: [Int], contractIndices: [Int]) -> MatrixDenseBLAS<S> {
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

    fileprivate func matricizedRightTensor(freeIndices: [Int], contractIndices: [Int]) -> MatrixDenseBLAS<S> {
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

    private func areStorageOrdered(_ first: [Int], _ second: [Int], rank: Int) -> Bool {
        var expected = 0
        for index in first {
            if index != expected { return false }
            expected += 1
        }
        for index in second {
            if index != expected { return false }
            expected += 1
        }
        return expected == rank
    }
}

private func doubleSum(_ left: [Double], _ right: [Double]) -> [Double] {
    #if canImport(Accelerate)
    return AccelerateOperations.add(left, right)
    #else
    var result = Array(repeating: 0.0, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
    #endif
}

private func floatSum(_ left: [Float], _ right: [Float]) -> [Float] {
    #if canImport(Accelerate)
    return AccelerateOperations.add(left, right)
    #else
    var result = Array(repeating: Float.zero, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
    #endif
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
    #if canImport(Accelerate)
    return AccelerateOperations.scale(values, by: scalar)
    #else
    var result = Array(repeating: 0.0, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
    #endif
}

private func floatScale(_ values: [Float], by scalar: Float) -> [Float] {
    #if canImport(Accelerate)
    return AccelerateOperations.scale(values, by: scalar)
    #else
    var result = Array(repeating: Float.zero, count: values.count)
    for index in 0..<values.count { result[index] = values[index] * scalar }
    return result
    #endif
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
