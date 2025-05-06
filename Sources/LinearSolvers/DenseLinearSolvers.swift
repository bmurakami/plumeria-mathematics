enum DenseRealLinearSolverDefault {
    @MainActor
    static var solver: DenseRealLinearSolver.Type = DenseRealLinearSolver_Accelerate.self
}

protocol DenseRealLinearSolver {
    static func solve(A: [[Double]], b: [Double]) throws -> [Double]
}

protocol DenseRealLinearSolver_LAPACK: DenseRealLinearSolver {}

extension DenseRealLinearSolver_LAPACK {
    static func lapack_dgesv(
        a: inout [Double],
        b: inout [Double],
        n: Int,
        solver: (
            _ n: UnsafeMutablePointer<Int32>,
            _ nrhs: UnsafeMutablePointer<Int32>,
            _ a: UnsafeMutablePointer<Double>,
            _ lda: UnsafeMutablePointer<Int32>,
            _ ipiv: UnsafeMutablePointer<Int32>,
            _ b: UnsafeMutablePointer<Double>,
            _ ldb: UnsafeMutablePointer<Int32>,
            _ info: UnsafeMutablePointer<Int32>
        ) -> Void
    ) throws {
        var n32 = Int32(n)
        var nrhs = Int32(1)
        var lda = Int32(n)
        var ldb = Int32(n)
        var info: Int32 = 0
        var ipiv = [Int32](repeating: 0, count: Int(n))
        
        solver(&n32, &nrhs, &a, &lda, &ipiv, &b, &ldb, &info)
        try checkLAPACKError(info: info)
        
        func checkLAPACKError(info: Int32) throws {
            if info < 0 { throw LAPACKError.lapackError(code: Int(info)) }
            else if info > 0 { throw LAPACKError.singularMatrix(message: "Matrix is singular") }
        }
    }
    
    static func validate(A: [[Double]], n: Int) throws {
        guard A.count == n
        else {
            throw LAPACKError.malformedProblem(message: "Matrix A must have \(n) rows to match vector b")
        }
        for row in A {
            guard row.count == n
            else {
                throw LAPACKError.malformedProblem(message: "Matrix A must be square (\(n)×\(n))")
            }
        }
    }
    
    static func convertToColumnMajor(A: [[Double]], n: Int) -> [Double] {
        var AT = [Double](repeating: 0.0, count: n * n)
        for i in 0..<n {
            for j in 0..<n { AT[j * n + i] = A[i][j] }
        }
        return AT
    }
}

enum LAPACKError: Error {
    case malformedProblem(message: String)
    case singularMatrix(message: String)
    case lapackError(code: Int)
}
