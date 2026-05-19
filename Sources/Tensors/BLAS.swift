#if canImport(Accelerate)
import AccelerateWrapper
#endif
import OpenBLASWrapper

public enum BLAS {
    #if canImport(Accelerate)
    case accelerate
    #endif
    case openBLAS

    public static var `default`: BLAS {
        #if canImport(Accelerate)
        return .accelerate
        #else
        return .openBLAS
        #endif
    }

    static func axpy(_ n: Int32, _ x: [Double], _ y: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.daxpy(n, x, &y)
        #else
        OpenBLASOperations.daxpy(n, x, &y)
        #endif
    }

    static func axpy(_ n: Int32, _ x: [Float], _ y: inout [Float]) {
        #if canImport(Accelerate)
        AccelerateOperations.saxpy(n, x, &y)
        #else
        OpenBLASOperations.saxpy(n, x, &y)
        #endif
    }

    static func scal(_ n: Int32, _ alpha: Double, _ x: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.dscal(n, alpha, &x)
        #else
        OpenBLASOperations.dscal(n, alpha, &x)
        #endif
    }

    static func scal(_ n: Int32, _ alpha: Float, _ x: inout [Float]) {
        #if canImport(Accelerate)
        AccelerateOperations.sscal(n, alpha, &x)
        #else
        OpenBLASOperations.sscal(n, alpha, &x)
        #endif
    }

    static func zaxpy(_ n: Int32, _ x: inout [Double], _ y: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.zaxpy(n, &x, &y)
        #else
        OpenBLASOperations.zaxpy(n, &x, &y)
        #endif
    }

    static func zscal(_ n: Int32, _ alpha: inout [Double], _ x: inout [Double]) {
        #if canImport(Accelerate)
        AccelerateOperations.zscal(n, &alpha, &x)
        #else
        OpenBLASOperations.zscal(n, &alpha, &x)
        #endif
    }
}
