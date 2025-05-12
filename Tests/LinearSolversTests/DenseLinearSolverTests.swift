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
    let v_actual = solveDenseRealLinear(A, b)
    let v_expected = DenseVector([1.0, 2, 3])
    
    #expect(v_actual.approximatelyEquals(v_expected))
}

@Test func solveDenseRealLinear_correctness_smallMatrices() throws {
    let sizes = [2, 3, 10]
    for n in sizes {
        let A = try makeMatrix(size: n)
        let b = makeVector(size: n)
        let v = solveDenseRealLinear(A, b)
        
        #expect(try (A • v as! DenseVector<Double>).approximatelyEquals(b, tolerance: 1e-13))
    }
}

@Test func solveDenseRealLinear_correctness_largeMatrices() throws {
    let sizes = [100, 1000]
    for n in sizes {
        let A = try makeMatrix(size: n)
        let b = makeVector(size: n)
        let v = solveDenseRealLinear(A, b)
        
        // LAPACK's dgesv seems to require more tolerance for large matrices.
        #expect(try (A • v as! DenseVector<Double>).approximatelyEquals(b, tolerance: 1e-7))
    }
}

func makeVector(size: Int) -> DenseVector<Double> {
    var v = [Double]()
    for _ in 0..<size {
        v.append(Double.random(in: -100.0...100.0))
    }
    return DenseVector(v)
}

func makeMatrix(size: Int) throws -> DenseMatrix<Double> {
    var matrix = [[Double]]()
    for _ in 0..<size {
        var row = [Double]()
        for _ in 0..<size {
            row.append(Double.random(in: -100.0...100.0))
        }
        matrix.append(row)
    }
    return try DenseMatrix(matrix)
}
