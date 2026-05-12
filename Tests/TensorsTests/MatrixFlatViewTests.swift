import Testing
@testable import Tensors

@Test func MatrixFlatView_wrapsRankTwoTensorFlatView() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0]])
    
    #expect(matrix.rows == 2)
    #expect(matrix.columns == 3)
    #expect(matrix.shape == [2, 3])
    #expect(matrix.rank == 2)
    #expect(matrix.elements == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(matrix.toArray() == [[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]])
    #expect(matrix.isContiguous)
}

@Test func MatrixFlatView_readsAndWritesElements() {
    var matrix = MatrixFlatView<Double>([[1.0, 2.0],
                                         [3.0, 4.0]])
    matrix[1, 0] = 99.0
    
    #expect(matrix[1, 0] == 99.0)
    #expect(matrix.toArray() == [[1.0, 2.0], [99.0, 4.0]])
}

@Test func MatrixFlatView_slicesRowsAndColumnsWithoutCopying() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]])
    let slice = matrix.slice(rows: SliceRange(1..<3), columns: SliceRange(1..<3))
    
    #expect(slice.view.storage === matrix.view.storage)
    #expect(slice.shape == [2, 2])
    #expect(slice.toArray() == [[6.0, 7.0],
                                [10.0, 11.0]])
    #expect(!slice.isContiguous)
}

@Test func MatrixFlatView_slicesRowToVectorFlatView() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]])
    let row: VectorFlatView<Double> = matrix.slice(row: 1, columns: SliceRange(0..<4, step: 2))
    
    #expect(row.view.storage === matrix.view.storage)
    #expect(row.shape == [2])
    #expect(row.elements == [5.0, 7.0])
    #expect(!row.isContiguous)
}

@Test func MatrixFlatView_slicesColumnToVectorFlatView() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0],
                                         [7.0, 8.0, 9.0]])
    let column: VectorFlatView<Double> = matrix.slice(rows: SliceRange(0..<3), column: 1)
    
    #expect(column.view.storage === matrix.view.storage)
    #expect(column.shape == [3])
    #expect(column.elements == [2.0, 5.0, 8.0])
    #expect(column.isContiguous)
}

@Test func MatrixFlatView_mutatingSliceCopiesOnWrite() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0],
                                         [3.0, 4.0]])
    var slice = matrix.slice(rows: SliceRange(0..<2), columns: SliceRange(1..<2))
    
    slice[1, 0] = 99.0
    
    #expect(slice.view.storage !== matrix.view.storage)
    #expect(matrix.toArray() == [[1.0, 2.0],
                                 [3.0, 4.0]])
    #expect(slice.toArray() == [[2.0], [99.0]])
}

@Test func MatrixFlatView_subscriptSlicesRowsAndColumns() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]])
    let slice: MatrixFlatView<Double> = matrix[1..<3, 1..<3]
    
    #expect(slice.toArray() == [[6.0, 7.0], [10.0, 11.0]])
}

@Test func MatrixFlatView_subscriptSlicesRowToVector() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]])
    let row: VectorFlatView<Double> = matrix[1, step(0..<4, by: 2)]
    
    #expect(row.elements == [5.0, 7.0])
}

@Test func MatrixFlatView_subscriptSlicesColumnToVector() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0],
                                         [7.0, 8.0, 9.0]])
    let column: VectorFlatView<Double> = matrix[all, 1]
    
    #expect(column.elements == [2.0, 5.0, 8.0])
}

@Test func MatrixFlatView_subscriptSlicesWithAllAndStep() {
    let matrix = MatrixFlatView<Double>([[1.0, 2.0, 3.0, 4.0],
                                         [5.0, 6.0, 7.0, 8.0],
                                         [9.0, 10.0, 11.0, 12.0]])
    let slice: MatrixFlatView<Double> = matrix[all, step(0..<4, by: 2)]
    
    #expect(slice.toArray() == [[1.0, 3.0],
                                [5.0, 7.0],
                                [9.0, 11.0]])
}
