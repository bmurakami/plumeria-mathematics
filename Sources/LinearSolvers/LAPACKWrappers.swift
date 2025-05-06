import COpenBLAS

func solveDenseRealLinearSystemx(A: [[Double]], b: [Double]) throws -> [Double] {
    let n = b.count
    try validate(A: A, n: n)
    var AT = convertToColumnMajor(A: A, n: n)
    var bCopy = Array(b)
    try openblas_dgesv(a: &AT, b: &bCopy, n: n)

    return bCopy
    
    func validate(A: [[Double]], n: Int) throws {
        guard A.count == n
        else {
            throw LAPACKError.malformedProblem(message: "Matrix A must have \(n) rows to match vector b")
        }
        for row in A {
            guard row.count == n
            else {
                throw LAPACKError.malformedProblem(message: "Matrix A must be square (\(n)×\(n))")
            }}
    }
    
    func openblas_dgesv(a: inout [Double], b: inout [Double], n: Int) throws {
        var nCLPK = Int32(n)
        var nrhsCLPK = Int32(1)
        var ldaCLPK = Int32(n)
        var ldbCLPK = Int32(n)
        var infoCLPK: Int32 = 0
        var ipiv = [Int32](repeating: 0, count: n)

        dgesv_(&nCLPK, &nrhsCLPK, &a, &ldaCLPK, &ipiv, &b, &ldbCLPK, &infoCLPK)
        try checkLAPACKError(info: infoCLPK)
    }

    func checkLAPACKError(info: Int32) throws {
        if info < 0 { throw LAPACKError.lapackError(code: Int(info)) }
        else if info > 0 { throw LAPACKError.singularMatrix(message: "Matrix is singular") }
    }

}

private func convertToColumnMajor(A: [[Double]], n: Int) -> [Double] {
    var AT = [Double](repeating: 0.0, count: n * n)
    for i in 0..<n {
        for j in 0..<n { AT[j * n + i] = A[i][j] }
    }
    return AT
}
