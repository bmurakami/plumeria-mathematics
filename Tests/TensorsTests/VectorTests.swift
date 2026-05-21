import Testing
@testable import Tensors

let a = [1.2, -3.4]
let b = [-4.5, 5.6]

enum VectorImplementation: CaseIterable, CustomStringConvertible {
    case reference
    case blas

    var description: String {
        switch self {
        case .reference: "reference"
        case .blas: "blas"
        }
    }

    func checkDoubleInit() {
        switch self {
        case .reference: verifyDoubleInit(VectorDenseReference<Double>.self)
        case .blas: verifyDoubleInit(VectorDenseBLAS<Double>.self)
        }
    }

    func checkComplexInit() {
        switch self {
        case .reference: verifyComplexInit(VectorDenseReference<ComplexDouble>.self)
        case .blas: verifyComplexInit(VectorDenseBLAS<ComplexDouble>.self)
        }
    }

    func checkNestedArrayInitializer() {
        switch self {
        case .reference: verifyNestedArrayInitializer(VectorDenseReference<Double>.self)
        case .blas: verifyNestedArrayInitializer(VectorDenseBLAS<Double>.self)
        }
    }

    func checkPhysicsOperations() {
        switch self {
        case .reference: verifyPhysicsOperations(VectorDenseReference<Double>.self)
        case .blas: verifyPhysicsOperations(VectorDenseBLAS<Double>.self)
        }
    }

    func checkArithmetic() {
        switch self {
        case .reference: verifyArithmetic(VectorDenseReference<Double>.self)
        case .blas: verifyArithmetic(VectorDenseBLAS<Double>.self)
        }
    }

    func checkComplexArithmetic() {
        switch self {
        case .reference: verifyComplexArithmetic(VectorDenseReference<ComplexDouble>.self)
        case .blas: verifyComplexArithmetic(VectorDenseBLAS<ComplexDouble>.self)
        }
    }

    func checkToArray() {
        switch self {
        case .reference: verifyToArray(VectorDenseReference<Double>.self)
        case .blas: verifyToArray(VectorDenseBLAS<Double>.self)
        }
    }

    func checkTensorStructure() {
        switch self {
        case .reference: verifyTensorStructure(VectorDenseReference<Double>.self)
        case .blas: verifyTensorStructure(VectorDenseBLAS<Double>.self)
        }
    }

    func checkSliceAssignment() {
        switch self {
        case .reference: verifySliceAssignment(VectorDenseReference<Double>.self)
        case .blas: verifySliceAssignment(VectorDenseBLAS<Double>.self)
        }
    }
}

@Test(arguments: VectorImplementation.allCases)
func VectorDense_initDouble(implementation: VectorImplementation) { implementation.checkDoubleInit() }

@Test(arguments: VectorImplementation.allCases)
func VectorDense_initComplex(implementation: VectorImplementation) { implementation.checkComplexInit() }

@Test(arguments: VectorImplementation.allCases)
func VectorDense_nestedArrayInitializer(implementation: VectorImplementation) {
    implementation.checkNestedArrayInitializer()
}

@Test(arguments: VectorImplementation.allCases)
func VectorDense_physicsOperations(implementation: VectorImplementation) { implementation.checkPhysicsOperations() }

@Test(arguments: VectorImplementation.allCases)
func VectorDense_arithmetic(implementation: VectorImplementation) { implementation.checkArithmetic() }

@Test(arguments: VectorImplementation.allCases)
func VectorDense_complexArithmetic(implementation: VectorImplementation) { implementation.checkComplexArithmetic() }

@Test(arguments: VectorImplementation.allCases)
func VectorDense_toArray(implementation: VectorImplementation) { implementation.checkToArray() }

@Test(arguments: VectorImplementation.allCases)
func VectorDense_tensorStructure(implementation: VectorImplementation) { implementation.checkTensorStructure() }

@Test(arguments: VectorImplementation.allCases)
func VectorDense_sliceAssignment(implementation: VectorImplementation) { implementation.checkSliceAssignment() }

@Test func DefaultVector_aliasUsesRecommendedImplementation() {
    let vector = Vector<Double>([1.0, 2.0, 3.0])

    #expect(type(of: vector) == VectorDenseBLAS<Double>.self)
    #expect(vector.toArray() == [1.0, 2.0, 3.0])
}

private func verifyDoubleInit<V: PluVector>(_ type: V.Type) where V.S == Double {
    let v = V(a)
    #expect(v.size == a.count)
    #expect(v[0] == a[0])
    #expect(v[1] == a[1])
}

private func verifyComplexInit<V: PluVector>(_ type: V.Type) where V.S == ComplexDouble {
    let s = ComplexDouble(a[0], a[1])
    let t = ComplexDouble(b[0], b[1])
    let v = V([s, t])
    #expect(v.size == a.count)
    #expect(v[0].real == a[0])
    #expect(v[0].imaginary == a[1])
    #expect(v[0] == ComplexDouble(a[0], a[1]))
    #expect(v[1].real == b[0])
    #expect(v[1].imaginary == b[1])
    #expect(v[1] == ComplexDouble(b[0], b[1]))
}

private func verifyNestedArrayInitializer<V: PluVector>(_ type: V.Type) where V.S == Double {
    let values: TensorNestedArray<Double> = [1.0, -2.0, 3.0]
    let vector = V(values)
    #expect(vector.shape == [3])
    #expect(vector.rank == 1)
    #expect(vector.toArray() == [1.0, -2.0, 3.0])
}

private func verifySliceAssignment<V: PluVector>(_ type: V.Type) where V.S == Double {
    var vector = V([1.0, 2.0, 3.0, 4.0])
    vector[1..<3] = V([20.0, 30.0])
    vector[step(0..<4, by: 2)] = V([10.0, 40.0])

    #expect(vector.toArray() == [10.0, 20.0, 40.0, 4.0])
}

private func verifyPhysicsOperations<V: PluVector>(_ type: V.Type) where V.S == Double {
    let v = V([3.0, 4.0, 12.0])
    let w = V([2.0, -1.0, 3.0])

    #expect(v.magnitude() == 13.0)
    #expect(v.dot(w) == 38.0)
    #expect(v.cross(w).toArray() == [24.0, 15.0, -11.0])
}

private func verifyArithmetic<V: PluVector>(_ type: V.Type) where V.S == Double {
    let u = V(a)
    let v = V(b)
    #expect((u + v).toArray(round: true) == [-3.3, 2.2])
    #expect((u - v).toArray(round: true) == [5.7, -9.0])
    #expect((-u).toArray(round: true) == [-1.2, 3.4])
    #expect((u * 2.0).toArray(round: true) == [2.4, -6.8])
    #expect((u / 2.0).toArray(round: true) == [0.6, -1.7])
}

private func verifyComplexArithmetic<V: PluVector>(_ type: V.Type) where V.S == ComplexDouble {
    let u = V([ComplexDouble(1.0, -1.0), ComplexDouble(0.0, 2.0)])
    let v = V([ComplexDouble(2.0, 1.0), ComplexDouble(-3.0, 0.0)])

    #expect((u + v).toArray() == [ComplexDouble(3.0, 0.0), ComplexDouble(-3.0, 2.0)])
    #expect((u * ComplexDouble(0.0, 2.0)).toArray() == [ComplexDouble(2.0, 2.0), ComplexDouble(-4.0, 0.0)])
    #expect((u * ComplexDouble(2.0, 0.0)).toArray() == [ComplexDouble(2.0, -2.0), ComplexDouble(0.0, 4.0)])
}

private func verifyToArray<V: PluVector>(_ type: V.Type) where V.S == Double {
    let vector = V([1.2, 3.4, 5.6])
    #expect(vector.toArray() == [1.2, 3.4, 5.6])
}

private func verifyTensorStructure<V: PluVector>(_ type: V.Type) where V.S == Double {
    let vector = V([1.0, 2.0, 3.0])
    #expect(vector.shape == [3])
    #expect(vector.rank == 1)
}
