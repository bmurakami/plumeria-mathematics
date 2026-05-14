import COpenBLAS

public enum OpenBLASOperations {
    // MARK: - BLAS
    public static func dgemv(
        _ m: Int32, _ n: Int32,
        _ a: UnsafeMutablePointer<Double>,
        _ x: UnsafeMutablePointer<Double>,
        _ y: UnsafeMutablePointer<Double>
    ) {
        let alpha: Double = 1.0
        let beta = 0.0
        let lda = Int32(m)
        let incx = Int32(1)
        let incy = Int32(1)
        
        COpenBLAS.cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }

    public static func zgemv(
        _ m: Int32, _ n: Int32,
        _ a: inout [Double],
        _ x: inout [Double],
        _ y: inout [Double]
    ) {
        var alpha = [1.0, 0.0]
        var beta = [0.0, 0.0]
        let lda = Int32(m)
        let incx = Int32(1)
        let incy = Int32(1)

        alpha.withUnsafeMutableBufferPointer { alpha in
            beta.withUnsafeMutableBufferPointer { beta in
                a.withUnsafeMutableBufferPointer { a in
                    x.withUnsafeMutableBufferPointer { x in
                        y.withUnsafeMutableBufferPointer { y in
                            COpenBLAS.cblas_zgemv(
                                CblasColMajor, CblasNoTrans, m, n,
                                alpha.baseAddress, a.baseAddress, lda, x.baseAddress, incx,
                                beta.baseAddress, y.baseAddress, incy
                            )
                        }
                    }
                }
            }
        }
    }
    
    public static func dgemm(
        _ m: Int32, _ n: Int32, _ k: Int32,
        _ a: UnsafeMutablePointer<Double>,
        _ b: UnsafeMutablePointer<Double>,
        _ c: UnsafeMutablePointer<Double>
    ) {
        let alpha: Double = 1.0
        let beta = 0.0
        let lda = m
        let ldb = k
        let ldc = m
        COpenBLAS.cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    public static func zgemm(
        _ m: Int32, _ n: Int32, _ k: Int32,
        _ a: inout [Double],
        _ b: inout [Double],
        _ c: inout [Double]
    ) {
        var alpha = [1.0, 0.0]
        var beta = [0.0, 0.0]
        let lda = m
        let ldb = k
        let ldc = m

        alpha.withUnsafeMutableBufferPointer { alpha in
            beta.withUnsafeMutableBufferPointer { beta in
                a.withUnsafeMutableBufferPointer { a in
                    b.withUnsafeMutableBufferPointer { b in
                        c.withUnsafeMutableBufferPointer { c in
                            COpenBLAS.cblas_zgemm(
                                CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                                alpha.baseAddress, a.baseAddress, lda, b.baseAddress, ldb,
                                beta.baseAddress, c.baseAddress, ldc
                            )
                        }
                    }
                }
            }
        }
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
        
        COpenBLAS.dgesv_(&nMutable, &nrhs, a, &lda, &ipiv, b, &ldb, &info)
        return info
    }
}
