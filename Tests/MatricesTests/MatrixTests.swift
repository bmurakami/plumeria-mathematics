import Testing
@testable import Matrices

@Test func RealDenseMatrix_initializer_with_values() async throws {
    var m = Matrix([[1, 2], [3, 4]])
    #expect(m[0, 0] == 1)
    #expect(m[1, 0] == 3)
    #expect(m[0, 1] == 2)
    #expect(m[1, 1] == 4)
    
    m[1, 0] = 3.14
    #expect(m[1, 0] == 3.14)
}
