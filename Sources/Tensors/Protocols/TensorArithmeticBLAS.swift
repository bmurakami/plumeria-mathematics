import Numerics

public protocol TensorArithmeticBLAS: TensorArithmetic, TensorStructure where S: PluScalar {
    init(shape: [Int], initialValue: S)
    var elements: [S] { get set }
}

extension TensorArithmeticBLAS {
    public static func + (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.shape == rhs.shape, "Tensors must have the same shape")
        var result = lhs
        result.elements = sum(lhs.elements, rhs.elements)
        return result
    }

    public static prefix func - (operand: Self) -> Self {
        operand * -1
    }

    public static func * (tensor: Self, scalar: S) -> Self {
        var result = tensor
        result.elements = scaled(tensor.elements, by: scalar)
        return result
    }

    public static func * (scalar: S, tensor: Self) -> Self {
        tensor * scalar
    }

    public static func / (tensor: Self, scalar: S) -> Self {
        let one: S = 1
        return tensor * (one / scalar)
    }

    public func isClose(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        guard shape == other.shape else { return false }
        for (left, right) in zip(elements, other.elements) {
            if !left.isClose(to: right, relativeTolerance: relativeTolerance) { return false }
        }
        return true
    }

    private static func sum(_ left: [S], _ right: [S]) -> [S] {
        switch S.self {
        case is Double.Type:
            let x = right as! [Double]
            var y = left as! [Double]
            BLAS.axpy(Int32(y.count), x, &y)
            return y as! [S]
        case is Float.Type:
            let x = right as! [Float]
            var y = left as! [Float]
            BLAS.axpy(Int32(y.count), x, &y)
            return y as! [S]
        case is ComplexDouble.Type:
            return BLASComplexStorage.sum(left as! [ComplexDouble], right as! [ComplexDouble]) as! [S]
        case is ComplexFloat.Type:
            return BLASComplexStorage.sum(left as! [ComplexFloat], right as! [ComplexFloat]) as! [S]
        default:
            fatalError("Unsupported scalar type")
        }
    }

    private static func scaled(_ values: [S], by scalar: S) -> [S] {
        switch S.self {
        case is Double.Type:
            var result = values as! [Double]
            BLAS.scal(Int32(result.count), scalar as! Double, &result)
            return result as! [S]
        case is Float.Type:
            var result = values as! [Float]
            BLAS.scal(Int32(result.count), scalar as! Float, &result)
            return result as! [S]
        case is ComplexDouble.Type:
            var result = BLASComplexStorage.interleaved(values as! [ComplexDouble])
            var alpha = BLASComplexStorage.interleaved([scalar as! ComplexDouble])
            BLAS.zscal(Int32(values.count), &alpha, &result)
            return BLASComplexStorage.toComplex(result) as! [S]
        case is ComplexFloat.Type:
            var result = BLASComplexStorage.interleaved(values as! [ComplexFloat])
            var alpha = BLASComplexStorage.interleaved([scalar as! ComplexFloat])
            BLAS.cscal(Int32(values.count), &alpha, &result)
            return BLASComplexStorage.toComplex(result) as! [S]
        default:
            fatalError("Unsupported scalar type")
        }
    }

}
