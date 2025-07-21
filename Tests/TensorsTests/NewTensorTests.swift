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
        let vector = Vector(data)
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
        let vector = Vector(count: count, repeating: value)
        #expect(vector.shape == [count])
        #expect(vector.elements == expected)
        #expect(vector.rank == 1)
        #expect(vector.count == count)
    }
    
    @Test("Vector subscript access", arguments: [
        ([10.0, 20.0, 30.0], 0, 10.0),
        ([10.0, 20.0, 30.0], 1, 20.0),
        ([10.0, 20.0, 30.0], 2, 30.0)
    ])
    func testVectorSubscript(data: [TestScalar], index: Int, expected: TestScalar) {
        let vector = Vector(data)
        #expect(vector[index] == expected)
        #expect(vector[[index]] == expected)
    }
    
    @Test("Vector subscript mutation")
    func testVectorSubscriptMutation() {
        var vector = Vector([1.0, 2.0, 3.0])
        vector[1] = 99.0
        #expect(vector[1] == 99.0)
        #expect(vector.elements == [1.0, 99.0, 3.0])
    }
    
    @Test("Vector description format")
    func testVectorDescription() {
        let vector = Vector([1.0, 2.0, 3.0])
        #expect(vector.description == "[1.0, 2.0, 3.0]")
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
        let matrix = Matrix(rows: rows, columns: cols, data: data)
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
        let matrix = Matrix(rows: rows, columns: cols, repeating: value)
        #expect(matrix.shape == [rows, cols])
        #expect(matrix.elements == expected)
        #expect(matrix.rows == rows)
        #expect(matrix.columns == cols)
        #expect(matrix.rank == 2)
        #expect(matrix.count == rows * cols)
    }
    
    @Test("Matrix subscript access", arguments: [
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, 0, 0, 1.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, 0, 2, 3.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, 1, 1, 5.0)
    ])
    func testMatrixSubscript(data: [TestScalar], rows: Int, cols: Int, row: Int, col: Int, expected: TestScalar) {
        let matrix = Matrix(rows: rows, columns: cols, data: data)
        #expect(matrix[[row, col]] == expected)
    }
    
    @Test("Matrix subscript mutation")
    func testMatrixSubscriptMutation() {
        var matrix = Matrix(rows: 2, columns: 2, data: [1.0, 2.0, 3.0, 4.0])
        matrix[[0, 1]] = 99.0
        #expect(matrix[[0, 1]] == 99.0)
        #expect(matrix.elements == [1.0, 99.0, 3.0, 4.0])
    }
    
    @Test("Matrix description format")
    func testMatrixDescription() {
        let matrix = Matrix(rows: 2, columns: 3, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let expected = "[1.0, 2.0, 3.0]\n[4.0, 5.0, 6.0]"
        #expect(matrix.description == expected)
    }
}

@Suite("Tensor Tests")
struct TensorTests {
    
    @Test("Tensor initialization with shape and data", arguments: [
        ([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ([4], [10.0, 20.0, 30.0, 40.0])
    ])
    func testTensorInitWithData(shape: [Int], data: [TestScalar]) {
        let tensor = Tensor(shape: shape, data: data)
        #expect(tensor.shape == shape)
        #expect(tensor.elements == data)
        #expect(tensor.rank == shape.count)
        #expect(tensor.count == data.count)
    }
    
    @Test("Tensor initialization with repeating value", arguments: [
        ([2, 3], 5.0, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
        ([1, 1, 1], 42.0, [42.0]),
        ([3], 0.0, [0.0, 0.0, 0.0])
    ])
    func testTensorInitWithRepeating(shape: [Int], value: TestScalar, expected: [TestScalar]) {
        let tensor = Tensor(shape: shape, repeating: value)
        #expect(tensor.shape == shape)
        #expect(tensor.elements == expected)
        #expect(tensor.rank == shape.count)
        #expect(tensor.count == expected.count)
    }
    
    @Test("Tensor subscript access in 3D", arguments: [
        ([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0, 0, 0], 1.0),
        ([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0, 0, 1], 2.0),
        ([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 1], 8.0)
    ])
    func testTensorSubscript3D(shape: [Int], data: [TestScalar], indices: [Int], expected: TestScalar) {
        let tensor = Tensor(shape: shape, data: data)
        #expect(tensor[indices] == expected)
    }
    
    @Test("Tensor subscript mutation")
    func testTensorSubscriptMutation() {
        var tensor = Tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
        tensor[[0, 1]] = 99.0
        #expect(tensor[[0, 1]] == 99.0)
        #expect(tensor.elements == [1.0, 99.0, 3.0, 4.0])
    }
    
    @Test("Tensor description format")
    func testTensorDescription() {
        let tensor = Tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let expected = "Tensor of shape: [2, 3] and elements: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])"
        #expect(tensor.description == expected)
    }
}

@Suite("Error Handling Tests")
struct ErrorTests {
    
    // Note: Out-of-bounds and size mismatch errors use precondition()
    // and try! which crash the program rather than throwing catchable errors.
    // These would need to be changed to proper throwing functions to be testable.
    
    @Test("Error enum exists and has correct cases")
    func testErrorEnumExists() {
        // Just verify the error types exist and can be created
        let dimensionError = TensorError.incompatibleDimensions(got: 2, expected: 3)
        let boundsError = TensorError.indexOutOfBounds(dimension: 0, index: 5, bound: 3)
        
        // These are the error types that would be thrown if the code used
        // proper error handling instead of precondition/try!
        switch dimensionError {
        case .incompatibleDimensions(let got, let expected):
            #expect(got == 2)
            #expect(expected == 3)
        default:
            #expect(Bool(false), "Wrong error type")
        }
        
        switch boundsError {
        case .indexOutOfBounds(let dimension, let index, let bound):
            #expect(dimension == 0)
            #expect(index == 5)
            #expect(bound == 3)
        default:
            #expect(Bool(false), "Wrong error type")
        }
    }
}

@Suite("Protocol Conformance Tests")
struct ProtocolTests {
    
    @Test("FlatTensor protocol properties work correctly")
    func testFlatTensorProtocol() {
        let vector = Vector([1.0, 2.0, 3.0, 4.0])
        let matrix = Matrix(rows: 2, columns: 3, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        let tensor = Tensor(shape: [2, 2, 2], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        #expect(vector.rank == 1)
        #expect(vector.count == 4)
        
        #expect(matrix.rank == 2)
        #expect(matrix.count == 6)
        
        #expect(tensor.rank == 3)
        #expect(tensor.count == 8)
    }
    
    @Test("FlatTensor subscript access through protocol")
    func testFlatTensorSubscriptAccess() {
        let vector = Vector([10.0, 20.0, 30.0])
        #expect(vector[[0]] == 10.0)
        #expect(vector[[1]] == 20.0)
        
        let matrix = Matrix(rows: 2, columns: 2, data: [1.0, 2.0, 3.0, 4.0])
        #expect(matrix[[0, 1]] == 2.0)
        #expect(matrix[[1, 0]] == 3.0)
    }
}

@Suite("Edge Cases")
struct EdgeCaseTests {
    
    @Test("Empty vector")
    func testEmptyVector() {
        let vector = Vector<TestScalar>([])
        #expect(vector.shape == [0])
        #expect(vector.elements.isEmpty)
        #expect(vector.rank == 1)
        #expect(vector.count == 0)
    }
    
    @Test("Single element structures")
    func testSingleElementStructures() {
        let vector = Vector([42.0])
        let matrix = Matrix(rows: 1, columns: 1, data: [42.0])
        let tensor = Tensor(shape: [1], data: [42.0])
        
        #expect(vector.count == 1)
        #expect(vector[[0]] == 42.0)
        
        #expect(matrix.count == 1)
        #expect(matrix[[0, 0]] == 42.0)
        
        #expect(tensor.count == 1)
        #expect(tensor[[0]] == 42.0)
    }
    
    @Test("Large tensor dimensions")
    func testLargeTensorDimensions() {
        let tensor = Tensor(shape: [10, 10, 10], repeating: 1.0)
        #expect(tensor.rank == 3)
        #expect(tensor.count == 1000)
        #expect(tensor[[9, 9, 9]] == 1.0)
    }
}
