import Testing
@testable import plumeria_mathematics

@Test func example() async throws {
    let A = [
        [2.0, 3.0],
        [5.0, 1.0]
    ]

    let b = [8.0, 7.0]

    do {
        let solution = try solveDenseRealLinearSystem(A: A, b: b)
        print("Solution:")
        print("x = \(solution[0])")
        print("y = \(solution[1])")
    }
    catch LAPACKError.malformedProblem(let message) { print("Problem malformed: \(message)") }
    catch LAPACKError.singularMatrix(let message) { print("Singular matrix: \(message)") }
    catch LAPACKError.lapackError(let code) { print("LAPACK error: \(code)") }
    catch { print("Unexpected error: \(error)") }
}
