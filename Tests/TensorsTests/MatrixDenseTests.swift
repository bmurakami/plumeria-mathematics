import Testing
@testable import Tensors

enum MatrixImplementation: CaseIterable, CustomStringConvertible {
    case reference
    case blas

    var description: String {
        switch self {
        case .reference: "reference"
        case .blas: "blas"
        }
    }

    func checkInitializerWithValues() {
        switch self {
        case .reference: verifyInitializerWithValues(MatrixDenseReference<Double>.self)
        case .blas: verifyInitializerWithValues(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkInitializerWithRowsAndColumns() {
        switch self {
        case .reference: verifyInitializerWithRowsAndColumns(MatrixDenseReference<Double>.self)
        case .blas: verifyInitializerWithRowsAndColumns(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkNestedArrayInitializer() {
        switch self {
        case .reference: verifyNestedArrayInitializer(MatrixDenseReference<Double>.self)
        case .blas: verifyNestedArrayInitializer(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkVectorMultiplication() {
        switch self {
        case .reference: verifyVectorMultiplication(MatrixDenseReference<Double>.self)
        case .blas: verifyVectorMultiplication(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkComplexVectorMultiplication() {
        switch self {
        case .reference: verifyComplexVectorMultiplication(MatrixDenseReference<ComplexDouble>.self)
        case .blas: verifyComplexVectorMultiplication(MatrixDenseBLAS<ComplexDouble>.self)
        }
    }

    func checkMatrixMultiplication() {
        switch self {
        case .reference: verifyMatrixMultiplication(MatrixDenseReference<Double>.self)
        case .blas: verifyMatrixMultiplication(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkComplexMatrixMultiplication() {
        switch self {
        case .reference: verifyComplexMatrixMultiplication(MatrixDenseReference<ComplexDouble>.self)
        case .blas: verifyComplexMatrixMultiplication(MatrixDenseBLAS<ComplexDouble>.self)
        }
    }

    func checkTranspose() {
        switch self {
        case .reference: verifyTranspose(MatrixDenseReference<Double>.self)
        case .blas: verifyTranspose(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkFlatten() {
        switch self {
        case .reference: verifyFlatten(MatrixDenseReference<Double>.self)
        case .blas: verifyFlatten(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkToArray() {
        switch self {
        case .reference: verifyToArray(MatrixDenseReference<Double>.self)
        case .blas: verifyToArray(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkTensorStructure() {
        switch self {
        case .reference: verifyTensorStructure(MatrixDenseReference<Double>.self)
        case .blas: verifyTensorStructure(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkClose() {
        switch self {
        case .reference: verifyClose(MatrixDenseReference<Double>.self)
        case .blas: verifyClose(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkArithmetic() {
        switch self {
        case .reference: verifyArithmetic(MatrixDenseReference<Double>.self)
        case .blas: verifyArithmetic(MatrixDenseBLAS<Double>.self)
        }
    }

    func checkSliceAssignment() {
        switch self {
        case .reference: verifySliceAssignment(MatrixDenseReference<Double>.self)
        case .blas: verifySliceAssignment(MatrixDenseBLAS<Double>.self)
        }
    }
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_initializerWithValues(implementation: MatrixImplementation) {
    implementation.checkInitializerWithValues()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_initializerWithRowsAndColumns(implementation: MatrixImplementation) {
    implementation.checkInitializerWithRowsAndColumns()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_nestedArrayInitializer(implementation: MatrixImplementation) {
    implementation.checkNestedArrayInitializer()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_vectorMultiplication(implementation: MatrixImplementation) {
    implementation.checkVectorMultiplication()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_complexVectorMultiplication(implementation: MatrixImplementation) {
    implementation.checkComplexVectorMultiplication()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_matrixMultiplication(implementation: MatrixImplementation) {
    implementation.checkMatrixMultiplication()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_complexMatrixMultiplication(implementation: MatrixImplementation) {
    implementation.checkComplexMatrixMultiplication()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_transpose(implementation: MatrixImplementation) {
    implementation.checkTranspose()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_flatten(implementation: MatrixImplementation) {
    implementation.checkFlatten()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_toArray(implementation: MatrixImplementation) {
    implementation.checkToArray()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_tensorStructure(implementation: MatrixImplementation) {
    implementation.checkTensorStructure()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_close(implementation: MatrixImplementation) {
    implementation.checkClose()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_arithmetic(implementation: MatrixImplementation) {
    implementation.checkArithmetic()
}

@Test(arguments: MatrixImplementation.allCases)
func MatrixDense_sliceAssignment(implementation: MatrixImplementation) {
    implementation.checkSliceAssignment()
}

@Test func MatrixDense_sliceAssignmentReportsShapeMismatch() {
    #expect(sliceAssignmentShapeError(destination: [2, 2], replacement: [2, 1]) ==
            "Assigned slice shape [2, 1] must match destination slice shape [2, 2]")
}

@Test func DefaultMatrix_sliceAssignment() {
    var matrix = Matrix<Double>([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    matrix[0..<2, 1..<3] = Matrix<Double>([[20.0, 30.0], [50.0, 60.0]])

    #expect(matrix.toArray() == [[1.0, 20.0, 30.0], [4.0, 50.0, 60.0], [7.0, 8.0, 9.0]])
}

private func verifyInitializerWithValues<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    var matrix = M([[1.0, 2.0], [3.0, 4.0]])
    #expect(matrix[0, 0] == 1.0)
    #expect(matrix[1, 0] == 3.0)
    #expect(matrix[0, 1] == 2.0)
    #expect(matrix[1, 1] == 4.0)
    matrix[1, 0] = 3.14
    #expect(matrix[1, 0] == 3.14)
}

private func verifyInitializerWithRowsAndColumns<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    var matrix = M(rows: 2, columns: 3, initialValue: .zero)
    #expect(matrix.rows == 2)
    #expect(matrix.columns == 3)
    for row in 0..<matrix.rows {
        for column in 0..<matrix.columns {
            #expect(matrix[row, column] == 0)
        }
    }
    matrix[1, 2] = 3.14
    #expect(matrix[1, 2] == 3.14)
}

private func verifyNestedArrayInitializer<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let values: TensorNestedArray<Double> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let matrix = M(values)

    #expect(matrix.shape == [2, 3])
    #expect(matrix.rank == 2)
    #expect(matrix.toArray() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
}

private func verifySliceAssignment<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    var matrix = M([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    matrix[0..<2, 1..<3] = M([[20.0, 30.0], [50.0, 60.0]])
    matrix[all, 0..<1] = M([[10.0], [40.0], [70.0]])

    #expect(matrix.toArray() == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 8.0, 9.0]])
}

private func verifyVectorMultiplication<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let matrix = M([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let vector = VectorDenseReference<Double>([2.0, 3.0, 4.0])
    #expect((matrix * vector).toArray() == [20.0, 47.0])
}

private func verifyComplexVectorMultiplication<M: PluMatrix>(_ type: M.Type) where M.S == ComplexDouble {
    let matrix = complexTestMatrixA(M.self)
    let vector = VectorDenseReference<ComplexDouble>([ComplexDouble(1.0, 0.0), ComplexDouble(0.0, 1.0),
                                                      ComplexDouble(2.0, 0.0)])
    #expect(matrix * vector == VectorDenseReference<ComplexDouble>([ComplexDouble(1.0, 1.0), ComplexDouble(6.0, -3.0)]))
}

private func verifyMatrixMultiplication<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let left = M([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let right = M([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    #expect((left * right).toArray() == [[58.0, 64.0], [139.0, 154.0]])
}

private func verifyComplexMatrixMultiplication<M: PluMatrix>(_ type: M.Type) where M.S == ComplexDouble {
    let left = complexTestMatrixA(M.self)
    let right = M([[ComplexDouble(1.0, 0.0), ComplexDouble(0.0, 1.0)],
                   [ComplexDouble(2.0, -1.0), ComplexDouble(-1.0, 0.0)],
                   [ComplexDouble(0.0, 0.0), ComplexDouble(1.0, 1.0)]])
    #expect((left * right).toArray() == [[ComplexDouble(5.0, -1.0), ComplexDouble(-2.0, 0.0)],
                                         [ComplexDouble(2.0, 3.0), ComplexDouble(4.0, 3.0)]])
}

private func verifyTranspose<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let matrix = M([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    let transposed = matrix.transpose()
    for row in 0..<transposed.rows {
        for column in 0..<transposed.columns {
            #expect(transposed[row, column] == matrix[column, row])
        }
    }
}

private func verifyFlatten<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let matrix = M([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #expect(matrix.flatten(columnMajorOrder: true) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    #expect(matrix.flatten(columnMajorOrder: false) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

private func verifyToArray<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let matrix = M(values)
    #expect(matrix.toArray() == values)
}

private func verifyTensorStructure<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let matrix = M([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #expect(matrix.shape == [2, 3])
    #expect(matrix.rank == 2)
}

private func verifyClose<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let a = M([[1.0, 2.0], [3.0, 4.0]])
    let b = M([[1.0, 2.0], [3.0, 4.0 + 1e-14]])
    let c = M([[1.0, 2.0], [3.0, 5.0]])
    #expect(a.isClose(to: b, relativeTolerance: 1e-12))
    #expect(!a.isClose(to: c, relativeTolerance: 1e-12))
}

private func verifyArithmetic<M: PluMatrix>(_ type: M.Type) where M.S == Double {
    let left = M([[1.2, -3.4], [0.5, 2.0]])
    let right = M([[-4.5, 5.6], [1.5, -2.0]])
    #expect((left + right).toArray(round: true) == [[-3.3, 2.2], [2.0, 0.0]])
    #expect((left - right).toArray(round: true) == [[5.7, -9.0], [-1.0, 4.0]])
    #expect((-left).toArray(round: true) == [[-1.2, 3.4], [-0.5, -2.0]])
    #expect((left * 2.0).toArray(round: true) == [[2.4, -6.8], [1.0, 4.0]])
    #expect((left / 2.0).toArray(round: true) == [[0.6, -1.7], [0.25, 1.0]])
}

private func complexTestMatrixA<M: PluMatrix>(_ type: M.Type) -> M where M.S == ComplexDouble {
    M([[ComplexDouble(1.0, 1.0), ComplexDouble(2.0, 0.0), ComplexDouble(0.0, -1.0)],
       [ComplexDouble(3.0, 0.0), ComplexDouble(-1.0, 1.0), ComplexDouble(2.0, -1.0)]])
}
