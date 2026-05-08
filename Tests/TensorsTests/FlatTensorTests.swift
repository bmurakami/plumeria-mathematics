import Testing
@testable import Tensors

// Assuming Double conforms to PluScalar for testing
typealias TestScalar = Double

@Suite("Vector Tests")
struct VectorTests {
    
    @Test("Vector initialization with elements", arguments: [
        ([1.0, 2.0, 3.0], [3]),
        ([1.0], [1]),
        ([], [0])
    ])
    func testVectorInitWithElements(data: [TestScalar], expectedShape: [Int]) {
        let vector = VectorDenseReference(data)
        #expect(vector.shape == expectedShape)
        #expect(vector.elements == data)
        #expect(vector.rank == 1)
        #expect(vector.count == data.count)
    }
    
    @Test("Vector initialization with repeating value", arguments: [
        (3, 5.0, [5.0, 5.0, 5.0]),
        (1, 42.0, [42.0]),
        (0, 10.0, [])
    ])
    func testVectorInitWithRepeating(count: Int, value: TestScalar, expected: [TestScalar]) {
        let vector = VectorDenseReference(shape: [count], elements: expected)
        #expect(vector.shape == [count])
        #expect(vector.elements == expected)
        #expect(vector.rank == 1)
        #expect(vector.count == count)
    }
    
    @Test("Vector initialization with zeroes for shape")
    func testVectorInitWithShape() {
        let vector = VectorDenseReference<TestScalar>(shape: [3])
        #expect(vector.shape == [3])
        #expect(vector.elements == [0.0, 0.0, 0.0])
        #expect(vector.rank == 1)
    }
    
    @Test("Vector subscript access", arguments: [
        ([10.0, 20.0, 30.0], 0, 10.0),
        ([10.0, 20.0, 30.0], 1, 20.0),
        ([10.0, 20.0, 30.0], 2, 30.0)
    ])
    func testVectorSubscript(data: [TestScalar], index: Int, expected: TestScalar) {
        let vector = VectorDenseReference(data)
        #expect(vector[index] == expected)
        #expect(vector[[index]] == expected)
    }
    
    @Test("Vector subscript mutation")
    func testVectorSubscriptMutation() {
        var vector = VectorDenseReference([1.0, 2.0, 3.0])
        vector[1] = 99.0
        #expect(vector[1] == 99.0)
        #expect(vector.elements == [1.0, 99.0, 3.0])
    }
    
}

@Suite("Matrix Tests")
struct MatrixTests {
    
    @Test("Matrix initialization with data", arguments: [
        (2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        (1, 4, [10.0, 20.0, 30.0, 40.0], [1, 4]),
        (3, 1, [5.0, 10.0, 15.0], [3, 1])
    ])
    func testMatrixInitWithData(rows: Int, cols: Int, data: [TestScalar], expectedShape: [Int]) {
        let matrix = FlatMatrix(rows: rows, columns: cols, data: data)
        #expect(matrix.shape == expectedShape)
        #expect(matrix.elements == data)
        #expect(matrix.rows == rows)
        #expect(matrix.columns == cols)
        #expect(matrix.rank == 2)
        #expect(matrix.count == rows * cols)
    }
    
    @Test("Matrix initialization with repeating value", arguments: [
        (2, 2, 7.0, [7.0, 7.0, 7.0, 7.0]),
        (1, 3, 0.0, [0.0, 0.0, 0.0]),
        (3, 1, -1.0, [-1.0, -1.0, -1.0])
    ])
    func testMatrixInitWithRepeating(rows: Int, cols: Int, value: TestScalar, expected: [TestScalar]) {
        let matrix = FlatMatrix(rows: rows, columns: cols, repeating: value)
        #expect(matrix.shape == [rows, cols])
        #expect(matrix.elements == expected)
        #expect(matrix.rows == rows)
        #expect(matrix.columns == cols)
        #expect(matrix.rank == 2)
        #expect(matrix.count == rows * cols)
    }
    
    @Test("Matrix initialization with zeroes for shape")
    func testMatrixInitWithShape() {
        let matrix = FlatMatrix<TestScalar>(shape: [2, 3])
        #expect(matrix.shape == [2, 3])
        #expect(matrix.elements == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #expect(matrix.rank == 2)
    }
    
    @Test("Matrix subscript access", arguments: [
        ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3, 0, 0, 1.0),
        ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3, 0, 2, 3.0),
        ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3, 1, 1, 5.0)
    ])
    func testMatrixSubscript(data: [TestScalar], rows: Int, cols: Int, row: Int, col: Int, expected: TestScalar) {
        let matrix = FlatMatrix(rows: rows, columns: cols, data: data)
        #expect(matrix[[row, col]] == expected)
    }
    
    @Test("Matrix subscript mutation")
    func testMatrixSubscriptMutation() {
        var matrix = FlatMatrix(rows: 2, columns: 2, data: [1.0, 3.0, 2.0, 4.0])
        matrix[[0, 1]] = 99.0
        #expect(matrix[[0, 1]] == 99.0)
        #expect(matrix.elements == [1.0, 3.0, 99.0, 4.0])
    }
    
    @Test("Matrix description format")
    func testMatrixDescription() {
        let matrix = FlatMatrix(rows: 2, columns: 3, data: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        let expected = "[1.0, 2.0, 3.0]\n[4.0, 5.0, 6.0]"
        #expect(matrix.description == expected)
    }
    
    @Test("Matrix flatten converts storage order")
    func testMatrixFlattenStorageOrder() {
        let matrix = FlatMatrix(rows: 2, columns: 3, data: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        #expect(matrix.flatten(order: .columnMajor) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        #expect(matrix.flatten(order: .rowMajor) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }
}

@Suite("Protocol Conformance Tests")
struct ProtocolTests {
    
    @Test("FlatTensor protocol properties work correctly")
    func testFlatTensorProtocol() {
        let vector = VectorDenseReference([1.0, 2.0, 3.0, 4.0])
        let matrix = FlatMatrix(rows: 2, columns: 3, data: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        
        #expect(vector.rank == 1)
        #expect(vector.count == 4)
        
        #expect(matrix.rank == 2)
        #expect(matrix.count == 6)
    }
    
    @Test("FlatTensor subscript access through protocol")
    func testFlatTensorSubscriptAccess() {
        let vector = VectorDenseReference([10.0, 20.0, 30.0])
        #expect(vector[[0]] == 10.0)
        #expect(vector[[1]] == 20.0)
        
        let matrix = FlatMatrix(rows: 2, columns: 2, data: [1.0, 3.0, 2.0, 4.0])
        #expect(matrix[[0, 1]] == 2.0)
        #expect(matrix[[1, 0]] == 3.0)
    }
}

@Suite("Edge Cases")
struct EdgeCaseTests {
    
    @Test("Empty vector")
    func testEmptyVector() {
        let vector = VectorDenseReference<TestScalar>([])
        #expect(vector.shape == [0])
        #expect(vector.elements.isEmpty)
        #expect(vector.rank == 1)
        #expect(vector.count == 0)
    }
    
    @Test("Single element structures")
    func testSingleElementStructures() {
        let vector = VectorDenseReference([42.0])
        let matrix = FlatMatrix(rows: 1, columns: 1, data: [42.0])
        
        #expect(vector.count == 1)
        #expect(vector[[0]] == 42.0)
        
        #expect(matrix.count == 1)
        #expect(matrix[[0, 0]] == 42.0)
    }
}
