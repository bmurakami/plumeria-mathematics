import Testing
@testable import Tensors

@Test func DenseMatrix_Reference_initializerWithValues() {
    var m = MatrixDenseReference<Double>([[1.0, 2.0], [3.0, 4.0]])
    #expect(m[0, 0] == 1.0)
    #expect(m[1, 0] == 3.0)
    #expect(m[0, 1] == 2.0)
    #expect(m[1, 1] == 4.0)
    
    m[1, 0] = 3.14
    #expect(m[1, 0] == 3.14)
}

@Test func DenseMatrix_Reference_initializerWithRowsAndColumns() {
    var m = MatrixDenseReference<Double>(rows: 2, columns: 3)
    
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

@Test func DenseMatrix_Reference_vectorMultiplication() {
    let A = MatrixDenseReference<Double>([[1.0, 2.0], [3.0, 4.0]])
    let v = VectorDenseReference<Double>([2.0, 3.0])
    let b = A * v

    #expect(b == VectorDenseReference<Double>([8.0, 18.0]))
}

@Test func DenseMatrix_Reference_matrixMultiplication() {
    let A = MatrixDenseReference<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let B = MatrixDenseReference<Double>([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    let C = A * B

    #expect(C.toArray() == [[58.0, 64.0], [139.0, 154.0]])
}

@Test func DenseMatrix_Reference_transpose() {
    let m = MatrixDenseReference<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let mt = m.transpose()
    for i in 0..<mt.rows {
        for j in 0..<mt.columns {
            #expect(mt[i, j] == m[j, i])
        }
    }
}

@Test func DenseMatrix_Reference_flatten() {
    let m = MatrixDenseReference<Double>.init([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    #expect(m.flatten(columnMajorOrder: true) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(m.flatten(columnMajorOrder: false) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

@Test func DenseMatrix_Reference_toArray() {
    let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let m = MatrixDenseReference<Double>(a)
    #expect(m.toArray() == a)
}

@Test func DenseMatrix_Reference_approximatelyEqual() {
    let a = MatrixDenseReference<Double>([[1.0, 2.0], [3.0, 4.0]])
    let b = MatrixDenseReference<Double>([[1.0, 2.0], [3.0, 4.0 + 1e-14]])
    let c = MatrixDenseReference<Double>([[1.0, 2.0], [3.0, 5.0]])
    
    #expect(a.isApproximatelyEqual(to: b, relativeTolerance: 1e-12))
    #expect(!a.isApproximatelyEqual(to: c, relativeTolerance: 1e-12))
}
