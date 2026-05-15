#if canImport(Accelerate)
import AccelerateWrapper
#endif
import OpenBLASWrapper

public protocol TensorArithmeticBLAS: TensorArithmetic, TensorStructure where S: PluScalar {
    init(shape: [Int], initialValue: S)
    var elements: [S] { get set }
}

extension TensorArithmeticBLAS where Magnitude == S.Magnitude {
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
        tensor * (1 / scalar)
    }

    public func isApproximatelyEqual(
        to other: Self,
        relativeTolerance: S.Magnitude = S.Magnitude.ulpOfOne.squareRoot(),
        norm: (Self) -> S.Magnitude = { _ in .zero }
    ) -> Bool {
        guard shape == other.shape else { return false }
        for (left, right) in zip(elements, other.elements) {
            if !left.isApproximatelyEqual(to: right, relativeTolerance: relativeTolerance) { return false }
        }
        return true
    }

    private static func sum(_ left: [S], _ right: [S]) -> [S] {
        switch S.self {
        case is Double.Type:
            var x = right as! [Double]
            var y = left as! [Double]
            axpy(Int32(y.count), &x, &y)
            return y as! [S]
        case is Complex.Type:
            var x = interleaved(right as! [Complex])
            var y = interleaved(left as! [Complex])
            zaxpy(Int32(right.count), &x, &y)
            return complexValues(y) as! [S]
        default:
            fatalError("Unsupported scalar type")
        }
    }

    private static func scaled(_ values: [S], by scalar: S) -> [S] {
        switch S.self {
        case is Double.Type:
            var result = values as! [Double]
            scal(Int32(result.count), scalar as! Double, &result)
            return result as! [S]
        case is Complex.Type:
            var result = interleaved(values as! [Complex])
            var alpha = interleaved([scalar as! Complex])
            zscal(Int32(values.count), &alpha, &result)
            return complexValues(result) as! [S]
        default:
            fatalError("Unsupported scalar type")
        }
    }

    private static func axpy(_ n: Int32, _ x: inout [Double], _ y: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.daxpy(n, &x, &y)
        #else
        OpenBLASOperations.daxpy(n, &x, &y)
        #endif
    }

    private static func scal(_ n: Int32, _ alpha: Double, _ x: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.dscal(n, alpha, &x)
        #else
        OpenBLASOperations.dscal(n, alpha, &x)
        #endif
    }

    private static func zaxpy(_ n: Int32, _ x: inout [Double], _ y: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.zaxpy(n, &x, &y)
        #else
        OpenBLASOperations.zaxpy(n, &x, &y)
        #endif
    }

    private static func zscal(_ n: Int32, _ alpha: inout [Double], _ x: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.zscal(n, &alpha, &x)
        #else
        OpenBLASOperations.zscal(n, &alpha, &x)
        #endif
    }

    private static func interleaved(_ values: [Complex]) -> [Double] {
        values.flatMap { [$0.real, $0.imaginary] }
    }

    private static func complexValues(_ values: [Double]) -> [Complex] {
        stride(from: 0, to: values.count, by: 2).map { Complex(values[$0], values[$0 + 1]) }
    }
}
