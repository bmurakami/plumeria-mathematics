import Testing
@testable import Tensors

@Test func MatrixDense_BLAS_accelerateComplexVectorMultiplication() {
    #if canImport(Accelerate)
    let A = complexTestMatrixA()
    let v = VectorDenseReference<ComplexDouble>([ComplexDouble(1.0, 0.0), ComplexDouble(0.0, 1.0),
                                                 ComplexDouble(2.0, 0.0)])
    let b = A * v

    #expect(b == VectorDenseReference<ComplexDouble>([ComplexDouble(1.0, 1.0), ComplexDouble(6.0, -3.0)]))

    var AAccelerate = A
    AAccelerate.blasImplementation = .accelerate
    #expect(AAccelerate * v == b)
    #endif
}

@Test func MatrixDense_BLAS_accelerateComplexMatrixMultiplication() {
    #if canImport(Accelerate)
    let A = complexTestMatrixA()
    let B = MatrixDenseBLAS<ComplexDouble>([[ComplexDouble(1.0, 0.0), ComplexDouble(0.0, 1.0)],
                                      [ComplexDouble(2.0, -1.0), ComplexDouble(-1.0, 0.0)],
                                      [ComplexDouble(0.0, 0.0), ComplexDouble(1.0, 1.0)]])
    let C = A * B

    #expect(C.toArray() == [[ComplexDouble(5.0, -1.0), ComplexDouble(-2.0, 0.0)],
                            [ComplexDouble(2.0, 3.0), ComplexDouble(4.0, 3.0)]])

    var AAccelerate = A
    AAccelerate.blasImplementation = .accelerate
    #expect((AAccelerate * B).toArray() == C.toArray())
    #endif
}

@Test func MatrixDense_BLAS_wholeMatrixAdditionMaterializes() {
    let left = MatrixDenseBLAS<Double>([[1.0, 2.0], [3.0, 4.0]])
    let right = MatrixDenseBLAS<Double>([[5.0, 6.0], [7.0, 8.0]])
    let result = left + right
    #expect(result.lazy == nil)
    #expect(result.toArray() == [[6.0, 8.0], [10.0, 12.0]])
}

@Test func MatrixDense_BLAS_sliceAdditionStaysLazy() {
    let left = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    let right = MatrixDenseBLAS<Double>([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    let result = left[0..<2, 0..<2] + right[0..<2, 0..<2]
    #expect(result.lazy != nil)
    #expect(result.toArray() == [[10.0, 10.0], [10.0, 10.0]])
}

@Test func MatrixDense_BLAS_lazyDuplicateTermsCancelOnMaterialization() {
    let a = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    let b = MatrixDenseBLAS<Double>([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    let aSlice = a[1..<3, 1..<3]
    let bSlice = b[1..<3, 1..<3]
    let result = aSlice + bSlice - bSlice

    #expect(result.lazy != nil)
    if let lazy = result.lazy {
        #expect(LazyMatrix<Double>.normalized(lazy.terms).count == 1)
    }
    #expect(result.toArray() == aSlice.toArray())
}

@Test func MatrixDense_BLAS_lazyDuplicateTermsCancelOnSliceAssignment() {
    let a = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    let b = MatrixDenseBLAS<Double>([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    var destination = MatrixDenseBLAS<Double>(rows: 3, columns: 3)

    destination[1..<3, 1..<3] = a[1..<3, 1..<3] + b[1..<3, 1..<3] - b[1..<3, 1..<3]

    #expect(destination.toArray() == [[0.0, 0.0, 0.0], [0.0, 5.0, 6.0], [0.0, 8.0, 9.0]])
}

@Test func MatrixDense_BLAS_shapeBasedAccess() {
    var m = MatrixDenseBLAS<Double>(shape: [2, 3], elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    #expect(m.shape == [2, 3])
    #expect(m.rank == 2)
    #expect(m.elements == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(m[0, 2] == 3.0)
    #expect(m[[1, 2]] == 6.0)

    m[[1, 0]] = 7.0
    #expect(m.elements == [1.0, 7.0, 2.0, 5.0, 3.0, 6.0])
}

@Test func MatrixDense_BLAS_copiesOnWrite() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])
    var copy = matrix

    copy[1, 0] = 99.0

    #expect(matrix[1, 0] == 4.0)
    #expect(copy[1, 0] == 99.0)
}

@Test func MatrixDense_BLAS_slicesRowsAndColumns() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [9.0, 10.0, 11.0, 12.0]])
    let slice = matrix.slice(rows: SliceRange(1..<3), columns: SliceRange(1..<3))

    #expect(slice.shape == [2, 2])
    #expect(slice.toArray() == [[6.0, 7.0], [10.0, 11.0]])
    #expect(slice.flatten(columnMajorOrder: true) == [6.0, 10.0, 7.0, 11.0])
    #expect(slice.flatten(columnMajorOrder: false) == [6.0, 7.0, 10.0, 11.0])
}

@Test func MatrixDense_BLAS_mutatingSliceCopiesOnWrite() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]])
    var slice = matrix.slice(rows: SliceRange(0..<2), columns: SliceRange(1..<2))

    slice[1, 0] = 99.0

    #expect(matrix[1, 1] == 5.0)
    #expect(slice[1, 0] == 99.0)
}

@Test func MatrixDense_BLAS_subscriptSlicesRowsAndColumns() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [9.0, 10.0, 11.0, 12.0]])
    let slice: MatrixDenseBLAS<Double> = matrix[1..<3, 1..<3]

    #expect(slice.toArray() == [[6.0, 7.0], [10.0, 11.0]])
}

@Test func MatrixDense_BLAS_subscriptSlicesRowToVector() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0, 4.0],
                                          [5.0, 6.0, 7.0, 8.0],
                                          [9.0, 10.0, 11.0, 12.0]])
    let r: VectorFlatView<Double> = matrix[1, all]

    #expect(r.elements == [5.0, 6.0, 7.0, 8.0])
}

@Test func MatrixDense_BLAS_subscriptSlicesColumnToVector() {
    let matrix = MatrixDenseBLAS<Double>([[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0],
                                          [7.0, 8.0, 9.0]])
    let c: VectorFlatView<Double> = matrix[all, 1]

    #expect(c.elements == [2.0, 5.0, 8.0])
}

private func complexTestMatrixA() -> MatrixDenseBLAS<ComplexDouble> {
    MatrixDenseBLAS<ComplexDouble>([[ComplexDouble(1.0, 1.0), ComplexDouble(2.0, 0.0), ComplexDouble(0.0, -1.0)],
                              [ComplexDouble(3.0, 0.0), ComplexDouble(-1.0, 1.0), ComplexDouble(2.0, -1.0)]])
}
