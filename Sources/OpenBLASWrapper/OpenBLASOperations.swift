import COpenBLAS

public struct OpenBLASOperations {
    public static func dgemv(
        _ m: Int32, _ n: Int32,
        _ a: UnsafeMutablePointer<Double>,
        _ x: UnsafeMutablePointer<Double>,
        _ y: UnsafeMutablePointer<Double>,
    ) {
        let alpha: Double = 1.0
        let beta = 0.0
        let lda = Int32(n)
        let incx = Int32(1)
        let incy = Int32(1)

        COpenBLAS.cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
}
