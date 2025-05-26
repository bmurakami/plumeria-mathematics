import Testing
@testable import Tensors

@Test func Complex_initializers() {
    let z = Complex(1.2, 3.4)
    #expect(z.re == 1.2)
    #expect(z.im == 3.4)
}

@Test func Complex_arithmetic() {
    let a = Complex(1.2, -2.3)
    let b = Complex(3.4, 5.6)
    
    #expect(a.re == 1.2)
    #expect(a.im == -2.3)
    #expect(-a == Complex(-1.2, 2.3))
    #expect(a+b == Complex(4.6, 3.3))
    #expect(a+b == b+a)
    #expect((a-b).approximatelyEquals(Complex(-2.2, -7.9)))
}

@Test func Double_approximatelyEquals_defaultTolerance() {
    let testCases = [
        (1e-16, true),
        (1e-15, true),
        (1e-14, false),
        (1e-13, false)
    ]
    let x = 1.0
    for testCase in testCases {
        let epsilon = testCase.0
        let y = 1.0 + epsilon
        #expect(x.approximatelyEquals(y) == testCase.1)
    }
}

@Test func Double_approximatelyEquals_tolerance() {
    let testCases = [
        (1e-5, true),
        (1e-4, true),
        (1e-3, false),
        (1e-2, false)
    ]
    let x = 1.0
    for testCase in testCases {
        let epsilon = testCase.0
        let y = 1.0 + epsilon
        #expect(x.approximatelyEquals(y, tolerance: 1e-4) == testCase.1)
    }
}
