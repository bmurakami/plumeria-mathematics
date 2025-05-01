import Accelerate

func solveDenseRealLinearSystem(A: [[Double]], b: [Double]) throws -> [Double] {
    let n = b.count
    try validate(A: A, n: n)
    var AT = convertToColumnMajor(A: A, n: n)
    var bCopy = Array(b)
    try lapack_dgesv(a: &AT, b: &bCopy, n: n)

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
    
    func lapack_dgesv(a: inout [Double], b: inout [Double], n: Int) throws {
        var nCLPK = __CLPK_integer(n)
        var nrhsCLPK = __CLPK_integer(1)
        var ldaCLPK = __CLPK_integer(n)
        var ldbCLPK = __CLPK_integer(n)
        var infoCLPK: __CLPK_integer = 0
        var ipiv = [__CLPK_integer](repeating: 0, count: n)

        dgesv_(&nCLPK, &nrhsCLPK, &a, &ldaCLPK, &ipiv, &b, &ldbCLPK, &infoCLPK)
        try checkLAPACKError(info: infoCLPK)
    }

    func checkLAPACKError(info: __CLPK_integer) throws {
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

enum LAPACKError: Error {
    case malformedProblem(message: String)
    case singularMatrix(message: String)
    case lapackError(code: Int)
}
