import Testing
@testable import Tensors

@Test
func RealDenseMatrix_initializerWithValues() throws {
    var m = try RealDenseMatrix([[1, 2], [3, 4]])
    #expect(m[0, 0] == 1)
    #expect(m[1, 0] == 3)
    #expect(m[0, 1] == 2)
    #expect(m[1, 1] == 4)
    
    m[1, 0] = 3.14
    #expect(m[1, 0] == 3.14)
}

@Test
func RealDenseMatrix_initializerWithRowsAndColumns() throws {
    var m = RealDenseMatrix(rows: 2, columns: 3)
    for i in 0..<m.rows {
        for j in 0..<m.columns {
            #expect(m[i, j] == 0)
        }
    }
    
    m[1, 2] = 3.14
    #expect(m[1, 2] == 3.14)
}

@Test
func RealDenseMatrix_initializerValidation() {
    let testCases = [
        [],
        [[]],
        [[1.0],[]],
        [[], [1.0]],
        [[1.0, 2.0], [3.0]],
        
    ]
    for badArray in testCases {
        #expect(throws: MatrixError.self) {
            try RealDenseMatrix(badArray)
        }
    }
}
