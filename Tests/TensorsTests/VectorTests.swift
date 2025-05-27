import Testing
@testable import Tensors

let a = [1.2, -3.4]
let b = [-4.5, 5.6]

@Test func VectorDenseReference_initDouble() {
    let v = VectorDenseReference<Double>(a)
    #expect(v.size == a.count)
    #expect(v[0] == a[0])
    #expect(v[1] == a[1])
}

@Test func VectorDenseReference_initComplex() {
    let s = Complex(a[0], a[1])
    let t = Complex(b[0], b[1])
    let v = VectorDenseReference<Complex>([s, t])
    #expect(v.size == a.count)
    #expect(v[0].re == a[0])
    #expect(v[0].im == a[1])
    #expect(v[0] == Complex(a[0], a[1]))
    #expect(v[1].re == b[0])
    #expect(v[1].im == b[1])
    #expect(v[1] == Complex(b[0], b[1]))
}

@Test func VectorDenseReference_negative() {
    let v = -VectorDenseReference<Double>(a)
    #expect(v.toArray() == [-1.2, 3.4])
}

@Test func VectorDenseReference_plus() {
    let u = VectorDenseReference<Double>(a)
    let v = VectorDenseReference<Double>(b)
    #expect((u + v).toArray(round: true) == [-3.3, 2.2])
}

@Test func VectorDenseReference_minus() {
    let u = VectorDenseReference<Double>(a)
    let v = VectorDenseReference<Double>(b)
    #expect((u - v).toArray(round: true) == [5.7, -9.0])
}

@Test func DenseVector_toArray() throws {
    let v1 = VectorDenseReference<Double>([1.2, 3.4, 5.6])
    #expect(v1.toArray() == [1.2, 3.4, 5.6])
}
