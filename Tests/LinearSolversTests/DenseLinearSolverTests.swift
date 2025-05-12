import Testing
import Tensors
@testable import LinearSolvers

@Test func solveDenseRealLinear_exampleUsage() throws {
    // Solution: x=1, y=2, z=3
    //
    // 2x - y + z = 3
    // x + 2y - z = 2
    // x + y - 4z = -9
    
    let A = try DenseMatrix([[2.0, -1, 1], [1, 2, -1], [1, 1, -4]])
    let b = DenseVector([3.0, 2, -9])
    let v = solveDenseRealLinear(A, b)
    
    let x = 5
}
