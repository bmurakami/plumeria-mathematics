import Testing
@testable import Tensors

@Test func Complex_init() {
    let z = Complex(1.2, 3.4)
    #expect(z.real == 1.2)
    #expect(z.imaginary == 3.4)
}

@Test func Complex_arithmetic() {
    let a = Complex(1.2, -2.3)
    let b = Complex(3.4, 5.6)
    
    #expect(a.real == 1.2)
    #expect(a.imaginary == -2.3)
    #expect(-a == Complex(-1.2, 2.3))
    #expect(a+b == Complex(4.6, 3.3))
    #expect(a+b == b+a)
    #expect((a-b).isApproximatelyEqual(to: Complex(-2.2, -7.9)))
}
