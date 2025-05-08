import Testing
@testable import Tensors

@Test
func RealDenseVector_initializerWithValues() throws {
    let v = try Vector([1.2, 3.4, 5.6])
    #expect(v.size == 3)
    #expect(v[0] == 1.2)
    #expect(v[1] == 3.4)
    #expect(v[2] == 5.6)
}

@Test
func RealDenseVector_initializerValidation() throws {
    let emptyVector: [Double] = []
    #expect(throws: MatrixError.self) {
        try Vector(emptyVector)
    }
}
