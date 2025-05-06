import Testing
import Foundation
@testable import LinearSolvers

func generateMatrix(_ size: Int) -> [[Double]] {
    srand48(42)
    
    var matrix = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
    for i in 0..<size {
        for j in 0..<size {
            matrix[i][j] = drand48() * 200 - 100
        }
    }
    return matrix
}

func generateArray(_ size: Int) -> [Double] {
    srand48(42)
    
    var array = [Double](repeating: 0, count: size)
    for i in 0..<size {
        array[i] = drand48() * 200 - 100
    }
    return array
}

@Test func AccelerateBenchmarkTest() async throws {
    let startTime = CFAbsoluteTimeGetCurrent()
    for i in 2..<300 {
        let A = generateMatrix(i)
        let v = generateArray(i)
        let solution = try DenseRealLinearSolver_Accelerate.solve(A: A, b: v)
    }
    let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
    print("AccelerateBenchmarkTest \(timeElapsed) seconds")
}

@Test func OpenBLASBenchmarkTest() async throws {
    let startTime = CFAbsoluteTimeGetCurrent()
    for i in 2..<400 {
        let A = generateMatrix(i)
        let v = generateArray(i)
        let solution = try DenseRealLinearSolver_OpenBLAS.solve(A: A, b: v)
    }
    let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
    print("OpenBLASBenchmarkTest \(timeElapsed) seconds")
}
