import Testing
import Tensors
@testable import LinearSolvers

@Test func solveDenseRealLinear_Accelerate_correctness() async throws {
    let A = try DenseMatrix([
        [2.0, 3.0],
        [5.0, 1.0]
    ])

    let b = DenseVector([8.0, 7.0])

        let solution = solveDenseRealLinear_Accelerate(A, b)
        print("Solution:")
        print("x = \(solution[0])")
        print("y = \(solution[1])")
}
