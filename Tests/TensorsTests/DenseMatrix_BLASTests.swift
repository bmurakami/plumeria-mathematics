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

    #expect(m.flatten(columnMajorOrder: true) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(m.flatten(columnMajorOrder: false) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

@Test func DenseMatrix_BLAS_toArray() {
    let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let m = MatrixDenseBLAS<Double>(a)
    #expect(m.toArray() == a)
}

@Test func DenseMatrix_BLAS_flatTensorConformance() {
    var m = MatrixDenseBLAS<Double>(shape: [2, 3], elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    
    #expect(m.shape == [2, 3])
    #expect(m.rank == 2)
    #expect(m.elements == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(m[0, 2] == 3.0)
    #expect(m[indices: 0, 2] == 3.0)
    #expect(m[[1, 2]] == 6.0)
    
    m[[1, 0]] = 7.0
    #expect(m.elements == [1.0, 7.0, 2.0, 5.0, 3.0, 6.0])
}

@Test func DenseMatrix_BLAS_view() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])
    let view = matrix.view()
    
    #expect(view.offset == 0)
    #expect(view.shape == [2, 3])
    #expect(view.strides == [1, 2])
    #expect(view.isContiguous)
    #expect(view.storage.elements == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(view[[0, 2]] == 3.0)
    #expect(view[[1, 2]] == 6.0)
}

@Test func DenseMatrix_BLAS_viewSlicesColumnsAndRows() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])
    let view = matrix.view()
    let column = view.slice(rows: SliceRange.all(length: matrix.rows), columns: SliceRange(1..<2))
    let row = view.slice(rows: SliceRange(1..<2), columns: SliceRange.all(length: matrix.columns))
    
    #expect(column.shape == [2, 1])
    #expect(column.strides == [1, 2])
    #expect(column.isContiguous)
    #expect(column[[0, 0]] == 2.0)
    #expect(column[[1, 0]] == 5.0)
    #expect(row.shape == [1, 3])
    #expect(row.strides == [1, 2])
    #expect(!row.isContiguous)
    #expect(row[[0, 0]] == 4.0)
    #expect(row[[0, 2]] == 6.0)
}
