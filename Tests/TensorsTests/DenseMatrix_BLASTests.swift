import Testing
@testable import Tensors

@Test func DenseMatrix_BLAS_initializerWithValues() {
    var m = MatrixDenseBLAS<Double>([[1.0, 2.0], [3.0, 4.0]])
    #expect(m[0, 0] == 1.0)
    #expect(m[1, 0] == 3.0)
    #expect(m[0, 1] == 2.0)
    #expect(m[1, 1] == 4.0)
    
    m[1, 0] = 3.14
    #expect(m[1, 0] == 3.14)
    
    #if canImport(Accelerate)
    m = MatrixDenseBLAS<Double>([[1.0, 2.0], [3.0, 4.0]])
    m.blasImplementation = .accelerate
    #expect(m[0, 0] == 1.0)
    #expect(m[1, 0] == 3.0)
    #expect(m[0, 1] == 2.0)
    #expect(m[1, 1] == 4.0)
    
    m[1, 0] = 3.14
    #expect(m[1, 0] == 3.14)
    #endif
}

@Test func DenseMatrix_BLAS_initializerWithRowsAndColumns() {
    var m = MatrixDenseBLAS<Double>.init(rows: 2, columns: 3)
    
    #expect(m.rows == 2)
    #expect(m.columns == 3)
    
    for i in 0..<m.rows {
        for j in 0..<m.columns {
            #expect(m[i, j] == 0)
        }
    }
    
    m[1, 2] = 3.14
    #expect(m[1, 2] == 3.14)
}

@Test func DenseMatrix_BLAS_vectorMultiplication() {
    let A = MatrixDenseBLAS<Double>([[1.0, 2.0],
                                     [3.0, 4.0]])
    let v = VectorDenseReference<Double>([2.0, 3.0])
    let b = A * v

    #expect(b == VectorDenseReference<Double>([8.0, 18.0]))
}

@Test func DenseMatrix_BLAS_transpose() {
    let m = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let mt = m.transpose()
    for i in 0..<mt.rows {
        for j in 0..<mt.columns {
            #expect(mt[i, j] == m[j, i])
        }
    }
}

@Test func DenseMatrix_BLAS_flatten() {
    let m = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    #expect(m.flatten(columnMajorOrder: true) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    #expect(m.flatten(columnMajorOrder: false) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

@Test func DenseMatrix_BLAS_toArray() {
    let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let m = MatrixDenseBLAS<Double>(a)
    #expect(m.toArray() == a)
}
