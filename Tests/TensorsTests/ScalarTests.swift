import Testing
@testable import Tensors

@Test func Complex_init() {
    let z = ComplexDouble(1.2, 3.4)
    #expect(z.real == 1.2)
    #expect(z.imaginary == 3.4)
}

@Test func Complex_arithmetic() {
    let a = ComplexDouble(1.2, -2.3)
    let b = ComplexDouble(3.4, 5.6)

    #expect(a.real == 1.2)
    #expect(a.imaginary == -2.3)
    #expect(-a == ComplexDouble(-1.2, 2.3))
    #expect(a+b == ComplexDouble(4.6, 3.3))
    #expect(a+b == b+a)
    #expect((a-b).isApproximatelyEqual(to: ComplexDouble(-2.2, -7.9)))
    #expect((a*b).isApproximatelyEqual(to: ComplexDouble(16.96, -1.1)))
    #expect((a/b).isApproximatelyEqual(to: ComplexDouble(-0.20503261882572227, -0.3387698042870456)))
}

@Test func Complex_mixedRealArithmetic() {
    let z = ComplexDouble(3.0, -2.0)

    #expect(5.0 + z == ComplexDouble(8.0, -2.0))
    #expect(z + 5.0 == ComplexDouble(8.0, -2.0))
    #expect(5.0 - z == ComplexDouble(2.0, 2.0))
    #expect(z - 5.0 == ComplexDouble(-2.0, -2.0))
    #expect(5.0 * z == ComplexDouble(15.0, -10.0))
    #expect(z * 5.0 == ComplexDouble(15.0, -10.0))
    #expect((5.0 / z).isApproximatelyEqual(to: ComplexDouble(15.0/13.0, 10.0/13.0)))
    #expect((z / 5.0).isApproximatelyEqual(to: ComplexDouble(0.6, -0.4)))
}

@Test func Complex_plumeriaScalarProperties() {
    let z = ComplexDouble(3.0, 4.0)

    #expect(ComplexDouble.i == ComplexDouble(0.0, 1.0))
    #expect(z.star == ComplexDouble(3.0, -4.0))
    #expect(z.mod == 5.0)
    #expect(z.arg.isApproximatelyEqual(to: 0.9272952180016122, relativeTolerance: 1e-15))
}

@Test func Complex_imaginaryUnitSupportsLocalMathNotation() {
    let i = ComplexDouble.i
    let z = 1.2 + 3.4 * i

    #expect(z == ComplexDouble(1.2, 3.4))
}

@Test func PluScalar_roundsWithPrecision() {
    #expect(1.23456.round() == 1.23456)
    #expect(1.23456.round(precision: 2) == 1.23)
    #expect(Float(1.2345678).round() == 1.234568)
    #expect(Float(1.2345678).round(precision: 2) == 1.23)
    #expect(ComplexDouble(1.23456, -2.34567).round(precision: 2) == ComplexDouble(1.23, -2.35))
}
