import Testing
@testable import Tensors

@Test func tensorCreation() {
    let tensor = Tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    #expect(tensor.shape == [2, 3])
    #expect(tensor.rank == 2)
    #expect(tensor.elementCount == 6)
}

@Test func tensorRepeatingInit() {
    let tensor = Tensor(shape: [2, 2], repeating: 5.0)
    #expect(tensor[0, 0] == 5.0)
    #expect(tensor[1, 1] == 5.0)
}

@Test func elementAccess() {
    let tensor = Tensor(shape: [2, 3, 4], data: Array(0..<24).map(Double.init))
    #expect(tensor[0, 0, 0] == 0.0)
    #expect(tensor[1, 2, 3] == 23.0)
    #expect(tensor[0, 1, 2] == 6.0)
}

@Test func elementModification() {
    var tensor = Tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
    tensor[0, 1] = 99.0
    #expect(tensor[0, 1] == 99.0)
    #expect(tensor[1, 0] == 3.0)
}

@Test func arrayIndexAccess() {
    let tensor = Tensor(shape: [2, 3], data: Array(1...6).map(Double.init))
    #expect(tensor[[0, 0]] == 1.0)
    #expect(tensor[[1, 2]] == 6.0)
}

@Test func slicing() {
    let tensor = Tensor(shape: [2, 3, 4], data: Array(0..<24).map(Double.init))
    
    let slice2D = tensor[1]
    #expect(slice2D.shape == [3, 4])
    #expect(slice2D[0, 0] == 12.0)
    #expect(slice2D[2, 3] == 23.0)
    
    let slice1D = tensor[1][1]
    #expect(slice1D.shape == [4])
    #expect(slice1D[0] == 16.0)
    #expect(slice1D[3] == 19.0)
}

@Test func chainedSlicing() {
    let tensor = Tensor(shape: [3, 3, 3], data: Array(0..<27).map(Double.init))
    
    let slice = tensor[2][0]
    #expect(slice.shape == [3])
    #expect(slice[0] == 18.0)
}

@Test func vectorOperations() {
    let vector = Tensor(shape: [5], data: [1.0, 2.0, 3.0, 4.0, 5.0])
    #expect(vector.rank == 1)
    
    // Slicing a 1D vector returns a scalar tensor (rank 0)
    let vectorSlice = vector[2]
    #expect(vectorSlice.shape == [])
    #expect(vectorSlice[[]] == 3.0)
}

@Test func elementAccessVersusSlicing() {
    let vector = Tensor(shape: [5], data: [1.0, 2.0, 3.0, 4.0, 5.0])
    
    // Element access using array subscript
    #expect(vector[[2]] == 3.0)
    
    // Slicing using single int subscript
    let slice = vector[2]
    #expect(slice.shape == [])
    #expect(slice[[]] == 3.0)
}

@Test func matrixOperations() {
    let matrix = Tensor(shape: [3, 4], data: Array(1...12).map(Double.init))
    
    let row = matrix[1]
    #expect(row.shape == [4])
    #expect(row[0] == 5.0)
    #expect(row[3] == 8.0)
}

@Test func scalarTensor() {
    let scalar = Tensor(shape: [], data: [42.0])
    #expect(scalar.rank == 0)
    #expect(scalar[[]] == 42.0)
}

//@Test func indexOutOfBoundsError() {
//    let tensor = Tensor(shape: [2, 3], data: Array(1...6).map(Double.init))
//    
//    #expect(throws: TensorError.self) {
//        _ = tensor[3, 0]
//    }
//    
//    #expect(throws: TensorError.self) {
//        _ = tensor[0, 5]
//    }
//}
//
//@Test func incompatibleDimensionsError() {
//    let tensor = Tensor(shape: [2, 3, 4], data: Array(0..<24).map(Double.init))
//    
//    #expect(throws: TensorError.self) {
//        _ = tensor[1, 2]
//    }
//    
//    #expect(throws: TensorError.self) {
//        _ = tensor[1, 2, 3, 4]
//    }
//}

@Test func strideCalculation() {
    let tensor = Tensor(shape: [2, 3, 4], data: Array(0..<24).map(Double.init))
    
    #expect(tensor[0, 0, 0] == 0.0)
    #expect(tensor[0, 0, 1] == 1.0)
    #expect(tensor[0, 1, 0] == 4.0)
    #expect(tensor[1, 0, 0] == 12.0)
}

@Test func copyOnWrite() {
    let original = Tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
    var copy = original
    
    copy[0, 0] = 99.0
    
    #expect(original[0, 0] == 1.0)
    #expect(copy[0, 0] == 99.0)
}

