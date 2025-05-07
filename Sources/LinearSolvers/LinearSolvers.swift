protocol LinearSolver {
    static func solve(A: [[Double]], b: [Double]) throws -> [Double]
}
