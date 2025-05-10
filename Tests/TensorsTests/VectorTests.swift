import Testing
@testable import Tensors

@Test
func RealDenseVector_initializerWithValues() throws {
    let v = DenseVector<Double>([1.2, 3.4, 5.6])
    #expect(v.count == 3)
    #expect(v[0] == 1.2)
    #expect(v[1] == 3.4)
    #expect(v[2] == 5.6)
}
