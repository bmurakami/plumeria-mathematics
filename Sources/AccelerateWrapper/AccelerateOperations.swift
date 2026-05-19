#if canImport(Accelerate)
import Accelerate

public struct AccelerateOperations {
    public static func sgemv(_ m: Int32, _ n: Int32, _ a: [Float], _ x: [Float], _ y: inout [Float]) {
        let alpha: Float = 1.0
        let beta: Float = 0.0
        let lda = Int32(m)
        let incx = Int32(1)
        let incy = Int32(1)
        a.withUnsafeBufferPointer { a in
            x.withUnsafeBufferPointer { x in
                y.withUnsafeMutableBufferPointer { y in
                    Accelerate.cblas_sgemv(
                        CblasColMajor, CblasNoTrans, m, n, alpha, a.baseAddress!, lda, x.baseAddress!, incx,
                        beta, y.baseAddress!, incy
                    )
                }
            }
        }
    }

    public static func dgemv(_ m: Int32, _ n: Int32, _ a: [Double], _ x: [Double], _ y: inout [Double]) {
        let alpha: Double = 1.0
        let beta = 0.0
        let lda = Int32(m)
        let incx = Int32(1)
        let incy = Int32(1)
        a.withUnsafeBufferPointer { a in
            x.withUnsafeBufferPointer { x in
                y.withUnsafeMutableBufferPointer { y in
                    Accelerate.cblas_dgemv(
                        CblasColMajor, CblasNoTrans, m, n, alpha, a.baseAddress!, lda, x.baseAddress!, incx,
                        beta, y.baseAddress!, incy
                    )
                }
            }
        }
    }

    public static func zgemv(_ m: Int32, _ n: Int32, _ a: inout [Double], _ x: inout [Double], _ y: inout [Double]) {
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

    public static func cgemv(_ m: Int32, _ n: Int32, _ a: inout [Float], _ x: inout [Float], _ y: inout [Float]) {
        var alpha: [Float] = [1.0, 0.0]
        var beta: [Float] = [0.0, 0.0]
        let lda = Int32(m)
        let incx = Int32(1)
        let incy = Int32(1)
        alpha.withUnsafeMutableBufferPointer { alpha in
            beta.withUnsafeMutableBufferPointer { beta in
                a.withUnsafeMutableBufferPointer { a in
                    x.withUnsafeMutableBufferPointer { x in
                        y.withUnsafeMutableBufferPointer { y in
                            Accelerate.cblas_cgemv(
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
        _ m: Int32, _ n: Int32, _ k: Int32, _ a: [Double], _ b: [Double], _ c: inout [Double]
    ) {
        let alpha: Double = 1.0
        let beta = 0.0
        let lda = m
        let ldb = k
        let ldc = m
        a.withUnsafeBufferPointer { a in
            b.withUnsafeBufferPointer { b in
                c.withUnsafeMutableBufferPointer { c in
                    Accelerate.cblas_dgemm(
                        CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.baseAddress!, lda,
                        b.baseAddress!, ldb, beta, c.baseAddress!, ldc
                    )
                }
            }
        }
    }

    public static func sgemm(
        _ m: Int32, _ n: Int32, _ k: Int32, _ a: [Float], _ b: [Float], _ c: inout [Float]
    ) {
        let alpha: Float = 1.0
        let beta: Float = 0.0
        let lda = m
        let ldb = k
        let ldc = m
        a.withUnsafeBufferPointer { a in
            b.withUnsafeBufferPointer { b in
                c.withUnsafeMutableBufferPointer { c in
                    Accelerate.cblas_sgemm(
                        CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.baseAddress!, lda,
                        b.baseAddress!, ldb, beta, c.baseAddress!, ldc
                    )
                }
            }
        }
    }

    public static func zgemm(
        _ m: Int32, _ n: Int32, _ k: Int32, _ a: inout [Double], _ b: inout [Double], _ c: inout [Double]
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

    public static func cgemm(
        _ m: Int32, _ n: Int32, _ k: Int32, _ a: inout [Float], _ b: inout [Float], _ c: inout [Float]
    ) {
        var alpha: [Float] = [1.0, 0.0]
        var beta: [Float] = [0.0, 0.0]
        let lda = m
        let ldb = k
        let ldc = m
        alpha.withUnsafeMutableBufferPointer { alpha in
            beta.withUnsafeMutableBufferPointer { beta in
                a.withUnsafeMutableBufferPointer { a in
                    b.withUnsafeMutableBufferPointer { b in
                        c.withUnsafeMutableBufferPointer { c in
                            Accelerate.cblas_cgemm(
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

    public static func daxpy(_ n: Int32, _ x: [Double], _ y: inout [Double]) {
        let alpha = 1.0
        let inc = Int32(1)
        x.withUnsafeBufferPointer { x in
            y.withUnsafeMutableBufferPointer { y in
                Accelerate.cblas_daxpy(n, alpha, x.baseAddress!, inc, y.baseAddress!, inc)
            }
        }
    }

    public static func saxpy(_ n: Int32, _ x: [Float], _ y: inout [Float]) {
        let alpha: Float = 1.0
        let inc = Int32(1)
        x.withUnsafeBufferPointer { x in
            y.withUnsafeMutableBufferPointer { y in
                Accelerate.cblas_saxpy(n, alpha, x.baseAddress!, inc, y.baseAddress!, inc)
            }
        }
    }

    public static func dscal(_ n: Int32, _ alpha: Double, _ x: inout [Double]) {
        let inc = Int32(1)
        Accelerate.cblas_dscal(n, alpha, &x, inc)
    }

    public static func sscal(_ n: Int32, _ alpha: Float, _ x: inout [Float]) {
        let inc = Int32(1)
        Accelerate.cblas_sscal(n, alpha, &x, inc)
    }

    public static func dnrm2(_ n: Int32, _ x: [Double]) -> Double {
        let inc = Int32(1)
        return x.withUnsafeBufferPointer { x in
            Accelerate.cblas_dnrm2(n, x.baseAddress!, inc)
        }
    }

    public static func snrm2(_ n: Int32, _ x: [Float]) -> Float {
        let inc = Int32(1)
        return x.withUnsafeBufferPointer { x in
            Accelerate.cblas_snrm2(n, x.baseAddress!, inc)
        }
    }

    public static func dznrm2(_ n: Int32, _ x: inout [Double]) -> Double {
        let inc = Int32(1)
        return x.withUnsafeMutableBufferPointer { x in
            Accelerate.cblas_dznrm2(n, OpaquePointer(UnsafeMutableRawPointer(x.baseAddress!)), inc)
        }
    }

    public static func scnrm2(_ n: Int32, _ x: inout [Float]) -> Float {
        let inc = Int32(1)
        return x.withUnsafeMutableBufferPointer { x in
            Accelerate.cblas_scnrm2(n, OpaquePointer(UnsafeMutableRawPointer(x.baseAddress!)), inc)
        }
    }

    public static func zaxpy(_ n: Int32, _ x: inout [Double], _ y: inout [Double]) {
        var alpha = [1.0, 0.0]
        let inc = Int32(1)
        alpha.withUnsafeMutableBufferPointer { alpha in
            x.withUnsafeMutableBufferPointer { x in
                y.withUnsafeMutableBufferPointer { y in
                    Accelerate.cblas_zaxpy(
                        n,
                        OpaquePointer(UnsafeMutableRawPointer(alpha.baseAddress!)),
                        OpaquePointer(UnsafeMutableRawPointer(x.baseAddress!)), inc,
                        OpaquePointer(UnsafeMutableRawPointer(y.baseAddress!)), inc
                    )
                }
            }
        }
    }

    public static func zscal(_ n: Int32, _ alpha: inout [Double], _ x: inout [Double]) {
        let inc = Int32(1)
        alpha.withUnsafeMutableBufferPointer { alpha in
            x.withUnsafeMutableBufferPointer { x in
                Accelerate.cblas_zscal(
                    n,
                    OpaquePointer(UnsafeMutableRawPointer(alpha.baseAddress!)),
                    OpaquePointer(UnsafeMutableRawPointer(x.baseAddress!)), inc
                )
            }
        }
    }

    public static func caxpy(_ n: Int32, _ x: inout [Float], _ y: inout [Float]) {
        var alpha: [Float] = [1.0, 0.0]
        let inc = Int32(1)
        alpha.withUnsafeMutableBufferPointer { alpha in
            x.withUnsafeMutableBufferPointer { x in
                y.withUnsafeMutableBufferPointer { y in
                    Accelerate.cblas_caxpy(
                        n,
                        OpaquePointer(UnsafeMutableRawPointer(alpha.baseAddress!)),
                        OpaquePointer(UnsafeMutableRawPointer(x.baseAddress!)), inc,
                        OpaquePointer(UnsafeMutableRawPointer(y.baseAddress!)), inc
                    )
                }
            }
        }
    }

    public static func cscal(_ n: Int32, _ alpha: inout [Float], _ x: inout [Float]) {
        let inc = Int32(1)
        alpha.withUnsafeMutableBufferPointer { alpha in
            x.withUnsafeMutableBufferPointer { x in
                Accelerate.cblas_cscal(
                    n,
                    OpaquePointer(UnsafeMutableRawPointer(alpha.baseAddress!)),
                    OpaquePointer(UnsafeMutableRawPointer(x.baseAddress!)), inc
                )
            }
        }
    }

    public static func dgesv(_ n: Int32, _ a: UnsafeMutablePointer<Double>, _ b: UnsafeMutablePointer<Double>)
        -> Int32 {
        var nMutable = n
        var nrhs = Int32(1)
        var lda = n
        var ipiv = Array<Int32>(repeating: 0, count: Int(n))
        var ldb = n
        var info = Int32(0)
        Accelerate.dgesv_(&nMutable, &nrhs, a, &lda, &ipiv, b, &ldb, &info)
        return info
    }

    public static func zgesv(_ n: Int32, _ a: inout [Double], _ b: inout [Double]) -> Int32 {
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
        _ n: Int32, _ a: inout [Double], _ wr: inout [Double], _ wi: inout [Double], _ vr: inout [Double]
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
