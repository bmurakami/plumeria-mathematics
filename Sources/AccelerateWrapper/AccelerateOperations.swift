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
        let lda = Int32(m)
        let incx = Int32(1)
        let incy = Int32(1)

        Accelerate.cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy)
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
                            Accelerate.cblas_zgemv(
                                CblasColMajor, CblasNoTrans, m, n,
                                OpaquePointer(UnsafeMutableRawPointer(alpha.baseAddress!)),
                                OpaquePointer(UnsafeMutableRawPointer(a.baseAddress!)), lda,
                                OpaquePointer(UnsafeMutableRawPointer(x.baseAddress!)), incx,
                                OpaquePointer(UnsafeMutableRawPointer(beta.baseAddress!)),
                                OpaquePointer(UnsafeMutableRawPointer(y.baseAddress!)), incy
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
        Accelerate.cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
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
                            Accelerate.cblas_zgemm(
                                CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                                OpaquePointer(UnsafeMutableRawPointer(alpha.baseAddress!)),
                                OpaquePointer(UnsafeMutableRawPointer(a.baseAddress!)), lda,
                                OpaquePointer(UnsafeMutableRawPointer(b.baseAddress!)), ldb,
                                OpaquePointer(UnsafeMutableRawPointer(beta.baseAddress!)),
                                OpaquePointer(UnsafeMutableRawPointer(c.baseAddress!)), ldc
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
        
        Accelerate.dgesv_(&nMutable, &nrhs, a, &lda, &ipiv, b, &ldb, &info)
        return info
    }

    public static func zgesv(
        _ n: Int32,
        _ a: inout [Double],
        _ b: inout [Double]
    ) -> Int32 {
        var nMutable = n
        var nrhs = Int32(1)
        var lda = n
        var ipiv = Array<Int32>(repeating: 0, count: Int(n))
        var ldb = n
        var info = Int32(0)

        a.withUnsafeMutableBufferPointer { a in
            b.withUnsafeMutableBufferPointer { b in
                Accelerate.zgesv_(
                    &nMutable, &nrhs,
                    OpaquePointer(UnsafeMutableRawPointer(a.baseAddress!)), &lda, &ipiv,
                    OpaquePointer(UnsafeMutableRawPointer(b.baseAddress!)), &ldb, &info
                )
            }
        }
        return info
    }

    public static func dgeev(
        _ n: Int32,
        _ a: inout [Double],
        _ wr: inout [Double],
        _ wi: inout [Double],
        _ vr: inout [Double]
    ) -> Int32 {
        var jobvl = Int8(UnicodeScalar("N").value)
        var jobvr = Int8(UnicodeScalar("V").value)
        var nMutable = n
        var lda = n
        var vl = Array(repeating: 0.0, count: 1)
        var ldvl = Int32(1)
        var ldvr = n
        var workQuery = 0.0
        var lwork = Int32(-1)
        var info = Int32(0)

        Accelerate.dgeev_(&jobvl, &jobvr, &nMutable, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr,
                          &workQuery, &lwork, &info)
        lwork = Int32(workQuery)
        var work = Array(repeating: 0.0, count: Int(lwork))
        Accelerate.dgeev_(&jobvl, &jobvr, &nMutable, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr,
                          &work, &lwork, &info)
        return info
    }
}
#endif
