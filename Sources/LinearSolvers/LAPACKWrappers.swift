import COpenBLAS

struct DenseRealLinearSolver_OpenBLAS: DenseRealLinearSolver_LAPACK {
    static func solve(A: [[Double]], b: [Double]) throws -> [Double] {
        let n = b.count
        try Self.validate(A: A, n: n)
        var AT = Self.convertToColumnMajor(A: A, n: n)
        var bCopy = Array(b)
        
        try Self.lapack_dgesv(a: &AT, b: &bCopy, n: n) { n, nrhs, a, lda, ipiv, b, ldb, info in
            COpenBLAS.dgesv_(n, nrhs, a, lda, ipiv, b, ldb, info)
        }
        
        return bCopy
    }
}
