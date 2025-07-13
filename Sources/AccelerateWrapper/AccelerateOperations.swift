#if canImport(Accelerate)
import Accelerate

public struct AccelerateOperations {
    public static func dgemv(
        _ m: Int32, _ n: Int32,
        _ a: UnsafeMutablePointer<Double>,
        _ x: UnsafeMutablePointer<Double>,
        _ y: UnsafeMutablePointer<Double>
    ) {
        let alpha: Double = 1.0
        let beta = 0.0
        let lda = Int32(n)
        let incx = Int32(1)
        let incy = Int32(1)

        Accelerate.cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    
    // MARK: - LAPACK
    public static func dgesv(
        _ n: Int32,
        _ a: UnsafeMutablePointer<Double>,
        _ b: UnsafeMutablePointer<Double>
    ) -> Int32 {
        var nMutable = n
        var nrhs = Int32(1)
        var lda = n
        var ipiv = Array<Int32>(repeating: 0, count: Int(n))
        var ldb = n
        var info = Int32(0)
        
        Accelerate.dgesv_(&nMutable, &nrhs, a, &lda, &ipiv, b, &ldb, &info)
        return info
    }
}
#endif
