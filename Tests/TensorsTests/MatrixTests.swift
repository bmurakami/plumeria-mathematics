import Testing
@testable import Tensors

@Test
func RealDenseMatrix_initializerWithValues() throws {
    var m = try DenseMatrix([[1.0, 2.0], [3.0, 4.0]])
    #expect(m[0, 0] == 1.0)
    #expect(m[1, 0] == 3.0)
    #expect(m[0, 1] == 2.0)
    #expect(m[1, 1] == 4.0)
    
    m[1, 0] = 3.14
    #expect(m[1, 0] == 3.14)
}

@Test
func RealDenseMatrix_initializerWithRowsAndColumns() throws {
    var m = DenseMatrix(rows: 2, columns: 3)
    
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

@Test
func RealDenseMatrix_transpose() throws {
    let m = try DenseMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let mt = m.t
    for i in 0..<mt.rows {
        for j in 0..<mt.columns {
            #expect(mt[i, j] as! Double == m[j, i])
        }
    }
}

@Test func RealDenseMatrix_flatten() throws {
    let m = try DenseMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #expect(m.flatten() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
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
            try DenseMatrix(badArray)
        }
    }
}
