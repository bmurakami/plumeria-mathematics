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

@Test func DenseMatrix_BLAS_rectangularVectorMultiplication() {
    let A = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]])
    let v = VectorDenseReference<Double>([2.0, 3.0, 4.0])
    let b = A * v

    #expect(b == VectorDenseReference<Double>([20.0, 47.0]))
}

@Test func DenseMatrix_BLAS_complexVectorMultiplication() {
    let A = MatrixDenseBLAS<Complex>([[Complex(1.0, 1.0), Complex(2.0, 0.0), Complex(0.0, -1.0)],
                                      [Complex(3.0, 0.0), Complex(-1.0, 1.0), Complex(2.0, -1.0)]])
    let v = VectorDenseReference<Complex>([Complex(1.0, 0.0), Complex(0.0, 1.0), Complex(2.0, 0.0)])
    let b = A * v

    #expect(b == VectorDenseReference<Complex>([Complex(1.0, 1.0), Complex(6.0, -3.0)]))
}

@Test func DenseMatrix_BLAS_matrixMultiplication() {
    let A = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let B = MatrixDenseBLAS<Double>([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    let C = A * B

    #expect(C.toArray() == [[58.0, 64.0], [139.0, 154.0]])
}

@Test func DenseMatrix_BLAS_complexMatrixMultiplication() {
    let A = MatrixDenseBLAS<Complex>([[Complex(1.0, 1.0), Complex(2.0, 0.0), Complex(0.0, -1.0)],
                                      [Complex(3.0, 0.0), Complex(-1.0, 1.0), Complex(2.0, -1.0)]])
    let B = MatrixDenseBLAS<Complex>([[Complex(1.0, 0.0), Complex(0.0, 1.0)],
                                      [Complex(2.0, -1.0), Complex(-1.0, 0.0)],
                                      [Complex(0.0, 0.0), Complex(1.0, 1.0)]])
    let C = A * B

    #expect(C.toArray() == [[Complex(5.0, -1.0), Complex(-2.0, 0.0)],
                            [Complex(2.0, 3.0), Complex(4.0, 3.0)]])
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

    #expect(m.flatten(columnMajorOrder: true) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(m.flatten(columnMajorOrder: false) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

@Test func DenseMatrix_BLAS_toArray() {
    let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let m = MatrixDenseBLAS<Double>(a)
    #expect(m.toArray() == a)
}

@Test func DenseMatrix_BLAS_shapeBasedAccess() {
    var m = MatrixDenseBLAS<Double>(shape: [2, 3], elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    
    #expect(m.shape == [2, 3])
    #expect(m.rank == 2)
    #expect(m.elements == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(m[0, 2] == 3.0)
    #expect(m[[1, 2]] == 6.0)
    
    m[[1, 0]] = 7.0
    #expect(m.elements == [1.0, 7.0, 2.0, 5.0, 3.0, 6.0])
}

@Test func DenseMatrix_BLAS_tensorStructure() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])

    #expect(matrix.shape == [2, 3])
    #expect(matrix.rank == 2)
}

@Test func DenseMatrix_BLAS_copiesOnWrite() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])
    var copy = matrix
    
    copy[1, 0] = 99.0
    
    #expect(matrix[1, 0] == 4.0)
    #expect(copy[1, 0] == 99.0)
}

@Test func DenseMatrix_BLAS_slicesRowsAndColumns() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [9.0, 10.0, 11.0, 12.0]])
    let slice = matrix.slice(rows: SliceRange(1..<3), columns: SliceRange(1..<3))
    
    #expect(slice.shape == [2, 2])
    #expect(slice.toArray() == [[6.0, 7.0], [10.0, 11.0]])
    #expect(slice.flatten(columnMajorOrder: true) == [6.0, 10.0, 7.0, 11.0])
    #expect(slice.flatten(columnMajorOrder: false) == [6.0, 7.0, 10.0, 11.0])
}

@Test func DenseMatrix_BLAS_mutatingSliceCopiesOnWrite() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])
    var slice = matrix.slice(rows: SliceRange(0..<2), columns: SliceRange(1..<2))
    
    slice[1, 0] = 99.0
    
    #expect(matrix[1, 1] == 5.0)
    #expect(slice[1, 0] == 99.0)
}

@Test func DenseMatrix_BLAS_subscriptSlicesRowsAndColumns() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [9.0, 10.0, 11.0, 12.0]])
    let slice: MatrixDenseBLAS<Double> = matrix[1..<3, 1..<3]
    
    #expect(slice.toArray() == [[6.0, 7.0], [10.0, 11.0]])
}

@Test func DenseMatrix_BLAS_subscriptSlicesRowToVector() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [9.0, 10.0, 11.0, 12.0]])
    let row: VectorFlatView<Double> = matrix[1, all]
    
    #expect(row.elements == [5.0, 6.0, 7.0, 8.0])
}

@Test func DenseMatrix_BLAS_subscriptSlicesColumnToVector() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0],
                                          [7.0, 8.0, 9.0]])
    let column: VectorFlatView<Double> = matrix[all, 1]
    
    #expect(column.elements == [2.0, 5.0, 8.0])
}
