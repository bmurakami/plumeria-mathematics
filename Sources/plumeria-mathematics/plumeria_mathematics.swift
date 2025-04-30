import Accelerate

func solveDenseRealLinearSystem(A: [[Double]], b: [Double]) throws -> [Double] {
    let n = b.count

    try validateProblem(A: A, n: n)
    var aColMajor = convertToColumnMajor(A: A, n: n)
    var bCopy = Array(b)

    try solveLapackSystem(a: &aColMajor, b: &bCopy, n: n)

    return bCopy
}

private func validateProblem(A: [[Double]], n: Int) throws {
    guard A.count == n else { throw LinAlgError.malformedProblem(message: "Matrix A must have \(n) rows to match vector b") }

    for row in A {
        guard row.count == n else { throw LinAlgError.malformedProblem(message: "Matrix A must be square (\(n)×\(n))") }
    }
}

private func convertToColumnMajor(A: [[Double]], n: Int) -> [Double] {
    var aColMajor = [Double](repeating: 0.0, count: n * n)

    for i in 0..<n {
        for j in 0..<n { aColMajor[j * n + i] = A[i][j] }
    }

    return aColMajor
}

private func solveLapackSystem(a: inout [Double], b: inout [Double], n: Int) throws {
    var nCLPK = __CLPK_integer(n)
    var nrhsCLPK = __CLPK_integer(1)
    var ldaCLPK = __CLPK_integer(n)
    var ldbCLPK = __CLPK_integer(n)
    var infoCLPK: __CLPK_integer = 0
    var ipiv = [__CLPK_integer](repeating: 0, count: n)

    dgesv_(&nCLPK, &nrhsCLPK, &a, &ldaCLPK, &ipiv, &b, &ldbCLPK, &infoCLPK)

    try checkLapackError(info: infoCLPK)
}

private func checkLapackError(info: __CLPK_integer) throws {
    if info < 0 { throw LinAlgError.lapackError(code: Int(info)) }
    else if info > 0 { throw LinAlgError.singularMatrix(message: "Matrix is singular") }
}

enum LinAlgError: Error {
    case malformedProblem(message: String)
    case singularMatrix(message: String)
    case lapackError(code: Int)
}
