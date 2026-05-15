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
    #expect((a*b).isApproximatelyEqual(to: Complex(16.96, -1.1)))
    #expect((a/b).isApproximatelyEqual(to: Complex(-0.20503261882572227, -0.3387698042870456)))
}

@Test func Complex_mixedRealArithmetic() {
    let z = Complex(3.0, -2.0)

    #expect(5.0 + z == Complex(8.0, -2.0))
    #expect(z + 5.0 == Complex(8.0, -2.0))
    #expect(5.0 - z == Complex(2.0, 2.0))
    #expect(z - 5.0 == Complex(-2.0, -2.0))
    #expect(5.0 * z == Complex(15.0, -10.0))
    #expect(z * 5.0 == Complex(15.0, -10.0))
    #expect((5.0 / z).isApproximatelyEqual(to: Complex(15.0/13.0, 10.0/13.0)))
    #expect((z / 5.0).isApproximatelyEqual(to: Complex(0.6, -0.4)))
}

@Test func Complex_plumeriaScalarProperties() {
    let z = Complex(3.0, 4.0)

    #expect(Complex.i == Complex(0.0, 1.0))
    #expect(z.star == Complex(3.0, -4.0))
    #expect(z.mod == 5.0)
    #expect(z.arg.isApproximatelyEqual(to: 0.9272952180016122, relativeTolerance: 1e-15))
}

@Test func Complex_imaginaryUnitSupportsLocalMathNotation() {
    let i = Complex.i
    let z = 1.2 + 3.4 * i

    #expect(z == Complex(1.2, 3.4))
}

@Test func PluScalar_roundsWithPrecision() {
    #expect(1.23456.round() == 1.23456)
    #expect(1.23456.round(precision: 2) == 1.23)
    #expect(Complex(1.23456, -2.34567).round(precision: 2) == Complex(1.23, -2.35))
}
