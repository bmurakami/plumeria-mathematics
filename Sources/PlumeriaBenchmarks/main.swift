import Foundation
import PlumeriaMathematics

struct TimedResult {
    let best: Double
    let median: Double
}

func measure(samples: Int = 5, iterations: Int = 1, operation body: () -> Double) -> (TimedResult, Double) {
    var checksum = body()
    var times: [Double] = []
    for _ in 0..<samples {
        let start = DispatchTime.now().uptimeNanoseconds
        var sampleChecksum = 0.0
        for _ in 0..<iterations { sampleChecksum += body() }
        let end = DispatchTime.now().uptimeNanoseconds
        checksum += sampleChecksum
        times.append(Double(end - start) / Double(iterations) / 1_000_000.0)
    }
    return (summarize(times), checksum)
}

func compareBenchmark(
    _ operation: String, _ size: String, samples: Int = 5, iterations: Int = 1,
    reference: () -> Double, blas: () -> Double
) -> Double {
    let (referenceResult, referenceChecksum) = measure(samples: samples, iterations: iterations, operation: reference)
    let (blasResult, blasChecksum) = measure(samples: samples, iterations: iterations, operation: blas)
    printBenchmark(operation, size, reference: referenceResult, blas: blasResult)
    return referenceChecksum + blasChecksum
}

func compareFloatBenchmark(
    _ operation: String, _ size: String, samples: Int = 5, iterations: Int = 1,
    double: () -> Double, float: () -> Double
) -> Double {
    let (doubleResult, doubleChecksum) = measure(samples: samples, iterations: iterations, operation: double)
    let (floatResult, floatChecksum) = measure(samples: samples, iterations: iterations, operation: float)
    printFloatBenchmark(operation, size, double: doubleResult, float: floatResult)
    return doubleChecksum + floatChecksum
}

func compareComplexFloatBenchmark(
    _ operation: String, _ size: String, samples: Int = 5, iterations: Int = 1,
    complex: () -> Double, complexFloat: () -> Double
) -> Double {
    let (complexResult, complexChecksum) = measure(samples: samples, iterations: iterations, operation: complex)
    let (complexFloatResult, complexFloatChecksum) = measure(
        samples: samples, iterations: iterations, operation: complexFloat
    )
    printComplexFloatBenchmark(operation, size, complex: complexResult, complexFloat: complexFloatResult)
    return complexChecksum + complexFloatChecksum
}

func compareImplementationBenchmark(
    _ operation: String, _ size: String, samples: Int = 3, iterations: Int = 1,
    accelerate: () -> Double, openBLAS: () -> Double
) -> Double {
    let (accelerateResult, accelerateChecksum) = measure(
        samples: samples, iterations: iterations, operation: accelerate
    )
    let (openBLASResult, openBLASChecksum) = measure(samples: samples, iterations: iterations, operation: openBLAS)
    printImplementationBenchmark(operation, size, accelerate: accelerateResult, openBLAS: openBLASResult)
    return accelerateChecksum + openBLASChecksum
}

func summarize(_ values: [Double]) -> TimedResult {
    let sorted = values.sorted()
    return TimedResult(best: sorted[0], median: sorted[sorted.count / 2])
}

func format(_ value: Double) -> String {
    String(format: "%.4f", value)
}

func printBenchmark(_ operation: String, _ size: String, reference: TimedResult, blas: TimedResult) {
    print("  \(operation) \(size)")
    print("    reference: median \(format(reference.median)) ms, best \(format(reference.best)) ms")
    print("    BLAS:      median \(format(blas.median)) ms, best \(format(blas.best)) ms")
    print("    ratio:     \(ratio(reference: reference.median, blas: blas.median))")
    print("")
}

func printFloatBenchmark(_ operation: String, _ size: String, double: TimedResult, float: TimedResult) {
    print("  \(operation) \(size)")
    print("    Double: median \(format(double.median)) ms, best \(format(double.best)) ms")
    print("    Float:  median \(format(float.median)) ms, best \(format(float.best)) ms")
    print("    ratio:  \(ratio(double: double.median, float: float.median))")
    print("")
}

func printComplexFloatBenchmark(_ operation: String, _ size: String, complex: TimedResult, complexFloat: TimedResult) {
    print("  \(operation) \(size)")
    print("    ComplexDouble: median \(format(complex.median)) ms, best \(format(complex.best)) ms")
    print("    ComplexFloat:  median \(format(complexFloat.median)) ms, best \(format(complexFloat.best)) ms")
    let speedRatio = ratio(left: complex.median, "ComplexDouble", right: complexFloat.median, "ComplexFloat")
    print("    ratio:         \(speedRatio)")
    print("")
}

func printImplementationBenchmark(_ operation: String, _ size: String, accelerate: TimedResult, openBLAS: TimedResult) {
    print("  \(operation) \(size)")
    print("    Accelerate: median \(format(accelerate.median)) ms, best \(format(accelerate.best)) ms")
    print("    OpenBLAS:   median \(format(openBLAS.median)) ms, best \(format(openBLAS.best)) ms")
    print("    ratio:      \(ratio(left: accelerate.median, "Accelerate", right: openBLAS.median, "OpenBLAS"))")
    print("")
}

func ratio(reference: Double, blas: Double) -> String {
    if reference == 0 && blas == 0 { return "same speed" }
    if reference == 0 { return "reference faster" }
    if blas == 0 { return "BLAS faster" }
    if reference < blas {
        return "reference \(format(blas / reference))x faster"
    }
    return "BLAS \(format(reference / blas))x faster"
}

func ratio(double: Double, float: Double) -> String {
    if double == 0 && float == 0 { return "same speed" }
    if double == 0 { return "Double faster" }
    if float == 0 { return "Float faster" }
    if double < float {
        return "Double \(format(float / double))x faster"
    }
    return "Float \(format(double / float))x faster"
}

func ratio(left: Double, _ leftLabel: String, right: Double, _ rightLabel: String) -> String {
    if left == 0 && right == 0 { return "same speed" }
    if left == 0 { return "\(leftLabel) faster" }
    if right == 0 { return "\(rightLabel) faster" }
    if left < right {
        return "\(leftLabel) \(format(right / left))x faster"
    }
    return "\(rightLabel) \(format(left / right))x faster"
}

func commandOutput(_ command: String, _ arguments: [String]) -> String {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: command)
    process.arguments = arguments
    process.standardOutput = pipe
    process.standardError = pipe
    do {
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.split(separator: "\n").first.map(String.init) ?? "unknown"
    } catch {
        return "unknown"
    }
}

func determinantChecksum(_ value: Double) -> Double {
    if value == 0.0 { return 0.0 }
    let sign = value < 0.0 ? -1.0 : 1.0
    if value.isInfinite { return sign * 308.0 }
    return sign * Foundation.log10(abs(value))
}

func writeBenchmarkResultToNullDevice(_ value: Double) {
    // Make benchmark results observable so the compiler does not release builds do not optimize the
    // calculations away by skipping them.
    guard let output = FileHandle(forWritingAtPath: "/dev/null") else { return }
    output.write(Data("\(value)\n".utf8))
}

func vectorValues(count: Int) -> [Double] {
    (0..<count).map { Double(($0 % 97) - 48) / 7.0 }
}

func vectorFloatValues(count: Int) -> [Float] {
    vectorValues(count: count).map(Float.init)
}

func vectorComplexValues(count: Int) -> [ComplexDouble] {
    vectorValues(count: count).map { ComplexDouble($0, -$0 / 2.0) }
}

func vectorComplexFloatValues(count: Int) -> [ComplexFloat] {
    vectorValues(count: count).map { ComplexFloat(Float($0), Float(-$0 / 2.0)) }
}

func matrixRows(rows: Int, columns: Int) -> [[Double]] {
    var values: [[Double]] = []
    values.reserveCapacity(rows)
    for row in 0..<rows {
        var rowValues: [Double] = []
        rowValues.reserveCapacity(columns)
        for column in 0..<columns {
            rowValues.append(Double(((row * 31 + column * 17) % 101) - 50) / 11.0)
        }
        values.append(rowValues)
    }
    return values
}

func invertibleMatrixRows(size: Int) -> [[Double]] {
    (0..<size).map { row in
        (0..<size).map { column in
            let value = Double(((row * 31 + column * 17) % 101) - 50) / 11.0
            return row == column ? value + Double(size) : value
        }
    }
}

func matrixFloatRows(rows: Int, columns: Int) -> [[Float]] {
    matrixRows(rows: rows, columns: columns).map { $0.map(Float.init) }
}

func matrixComplexRows(rows: Int, columns: Int) -> [[ComplexDouble]] {
    matrixRows(rows: rows, columns: columns).map { row in row.map { ComplexDouble($0, -$0 / 3.0) } }
}

func matrixComplexFloatRows(rows: Int, columns: Int) -> [[ComplexFloat]] {
    matrixRows(rows: rows, columns: columns).map { row in row.map { ComplexFloat(Float($0), Float(-$0 / 3.0)) } }
}

func fillTensor<T: TensorMultiplication>(_ tensor: inout T) where T.S == Double {
    for index in indexCombinations(for: tensor.shape) {
        let weighted = index.enumerated().reduce(0) { $0 + ($1.offset + 1) * ($1.element + 1) }
        tensor[index] = Double((weighted % 29) - 14) / 5.0
    }
}

func fillFloatTensor<T: TensorMultiplication>(_ tensor: inout T) where T.S == Float {
    for index in indexCombinations(for: tensor.shape) {
        let weighted = index.enumerated().reduce(0) { $0 + ($1.offset + 1) * ($1.element + 1) }
        tensor[index] = Float((weighted % 29) - 14) / 5.0
    }
}

func fillComplexTensor<T: TensorMultiplication>(_ tensor: inout T) where T.S == ComplexDouble {
    for index in indexCombinations(for: tensor.shape) {
        let weighted = index.enumerated().reduce(0) { $0 + ($1.offset + 1) * ($1.element + 1) }
        let value = Double((weighted % 29) - 14) / 5.0
        tensor[index] = ComplexDouble(value, -value / 2.0)
    }
}

func fillComplexFloatTensor<T: TensorMultiplication>(_ tensor: inout T) where T.S == ComplexFloat {
    for index in indexCombinations(for: tensor.shape) {
        let weighted = index.enumerated().reduce(0) { $0 + ($1.offset + 1) * ($1.element + 1) }
        let value = Float((weighted % 29) - 14) / 5.0
        tensor[index] = ComplexFloat(value, -value / 2.0)
    }
}

func indexCombinations(for shape: [Int]) -> [[Int]] {
    if shape.isEmpty { return [[]] }
    if shape.contains(0) { return [] }
    return (0..<shape.reduce(1, *)).map { flatIndex in
        var remaining = flatIndex
        return shape.map { dimension in
            let index = remaining % dimension
            remaining /= dimension
            return index
        }
    }
}

func benchmarkVectors() -> Double {
    print("Vector")
    let small = vectorValues(count: 10_000)
    let large = vectorValues(count: 100_000)
    let referenceSmall = VectorDenseReference(small)
    let referenceLarge = VectorDenseReference(large)
    let blasSmall = VectorDenseBLAS(small)
    let blasLarge = VectorDenseBLAS(large)
    var checksum = 0.0
    checksum += compareBenchmark("add", "10,000", iterations: 20, reference: {
        let result = referenceSmall + referenceSmall
        return result[0] + result[result.size - 1]
    }, blas: {
        let result = blasSmall + blasSmall
        return result[0] + result[result.size - 1]
    })
    checksum += compareBenchmark("scale", "100,000", iterations: 10, reference: {
        let result = referenceLarge * 1.25
        return result[0] + result[result.size - 1]
    }, blas: {
        let result = blasLarge * 1.25
        return result[0] + result[result.size - 1]
    })
    checksum += compareBenchmark("magnitude", "100,000", iterations: 10, reference: {
        referenceLarge.magnitude()
    }, blas: {
        blasLarge.magnitude()
    })
    return checksum
}

func benchmarkMatrices() -> Double {
    print("Matrix")
    let addRows = matrixRows(rows: 256, columns: 256)
    let largeAddRows = matrixRows(rows: 1_024, columns: 1_024)
    let vector = VectorDenseBLAS(vectorValues(count: 384))
    let largeVector = VectorDenseBLAS(vectorValues(count: 1_536))
    let mvRows = matrixRows(rows: 384, columns: 384)
    let largeMVRows = matrixRows(rows: 1_536, columns: 1_536)
    let mmLeftRows = matrixRows(rows: 128, columns: 128)
    let mmRightRows = matrixRows(rows: 128, columns: 128)
    let detRows = invertibleMatrixRows(size: 96)
    let referenceAdd = MatrixDenseReference(addRows)
    let blasAdd = MatrixDenseBLAS(addRows)
    let referenceLargeAdd = MatrixDenseReference(largeAddRows)
    let blasLargeAdd = MatrixDenseBLAS(largeAddRows)
    let referenceMV = MatrixDenseReference(mvRows)
    let blasMV = MatrixDenseBLAS(mvRows)
    let referenceLargeMV = MatrixDenseReference(largeMVRows)
    let blasLargeMV = MatrixDenseBLAS(largeMVRows)
    let referenceMMLeft = MatrixDenseReference(mmLeftRows)
    let referenceMMRight = MatrixDenseReference(mmRightRows)
    let blasMMLeft = MatrixDenseBLAS(mmLeftRows)
    let blasMMRight = MatrixDenseBLAS(mmRightRows)
    let referenceDet = MatrixDenseReference(detRows)
    let blasDet = MatrixDenseBLAS(detRows)
    var checksum = 0.0
    checksum += compareBenchmark("add", "256x256", iterations: 5, reference: {
        let result = referenceAdd + referenceAdd
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, blas: {
        let result = blasAdd + blasAdd
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareBenchmark("add", "1,024x1,024", samples: 3, reference: {
        let result = referenceLargeAdd + referenceLargeAdd
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, blas: {
        let result = blasLargeAdd + blasLargeAdd
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareBenchmark("matrix-vector multiply", "384x384", reference: {
        let result = referenceMV * vector
        return result[0] + result[result.size - 1]
    }, blas: {
        let result = blasMV * vector
        return result[0] + result[result.size - 1]
    })
    checksum += compareBenchmark("matrix-vector multiply", "1,536x1,536", samples: 3, reference: {
        let result = referenceLargeMV * largeVector
        return result[0] + result[result.size - 1]
    }, blas: {
        let result = blasLargeMV * largeVector
        return result[0] + result[result.size - 1]
    })
    checksum += compareBenchmark("matrix-matrix multiply", "128x128", reference: {
        let result = referenceMMLeft * referenceMMRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, blas: {
        let result = blasMMLeft * blasMMRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareBenchmark("determinant", "96x96", samples: 3, reference: {
        determinantChecksum(referenceDet.det)
    }, blas: {
        determinantChecksum(blasDet.det)
    })
    checksum += compareBenchmark("inverse", "96x96", samples: 3, reference: {
        let result = referenceDet.inverse()
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, blas: {
        let result = blasDet.inverse()
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    return checksum
}

func benchmarkTensors() -> Double {
    print("Tensor")
    var referenceAdd = TensorDenseReference<Double>(shape: [40, 40, 10], initialValue: 0.0)
    var blasAdd = TensorDenseBLAS<Double>(shape: [40, 40, 10], initialValue: 0.0)
    var referenceLeft = TensorDenseReference<Double>(shape: [16, 24, 16], initialValue: 0.0)
    var referenceRight = TensorDenseReference<Double>(shape: [16, 16], initialValue: 0.0)
    var blasLeft = TensorDenseBLAS<Double>(shape: [16, 24, 16], initialValue: 0.0)
    var blasRight = TensorDenseBLAS<Double>(shape: [16, 16], initialValue: 0.0)
    var referenceLargeLeft = TensorDenseReference<Double>(shape: [32, 32, 32], initialValue: 0.0)
    var referenceLargeRight = TensorDenseReference<Double>(shape: [32, 32], initialValue: 0.0)
    var blasLargeLeft = TensorDenseBLAS<Double>(shape: [32, 32, 32], initialValue: 0.0)
    var blasLargeRight = TensorDenseBLAS<Double>(shape: [32, 32], initialValue: 0.0)
    fillTensor(&referenceAdd)
    fillTensor(&blasAdd)
    fillTensor(&referenceLeft)
    fillTensor(&referenceRight)
    fillTensor(&blasLeft)
    fillTensor(&blasRight)
    fillTensor(&referenceLargeLeft)
    fillTensor(&referenceLargeRight)
    fillTensor(&blasLargeLeft)
    fillTensor(&blasLargeRight)
    var checksum = 0.0
    checksum += compareBenchmark("add", "40x40x10", iterations: 3, reference: {
        let result = referenceAdd + referenceAdd
        return result[[0, 0, 0]] + result[[39, 39, 9]]
    }, blas: {
        let result = blasAdd + blasAdd
        return result[[0, 0, 0]] + result[[39, 39, 9]]
    })
    checksum += compareBenchmark("contraction", "16x24x16,16x16", reference: {
        let result = multiply(referenceLeft, ["i", "j", "k"], referenceRight, ["k", "l"])
        return result[[0, 0, 0]] + result[[15, 23, 15]]
    }, blas: {
        let result = multiply(blasLeft, ["i", "j", "k"], blasRight, ["k", "l"])
        return result[[0, 0, 0]] + result[[15, 23, 15]]
    })
    checksum += compareBenchmark("contraction", "32x32x32,32x32", samples: 3, reference: {
        let result = multiply(referenceLargeLeft, ["i", "j", "k"], referenceLargeRight, ["k", "l"])
        return result[[0, 0, 0]] + result[[31, 31, 31]]
    }, blas: {
        let result = multiply(blasLargeLeft, ["i", "j", "k"], blasLargeRight, ["k", "l"])
        return result[[0, 0, 0]] + result[[31, 31, 31]]
    })
    return checksum
}

func benchmarkFloatScalars() -> Double {
    print("Float vs. Double")
    let doubleVector = VectorDenseBLAS(vectorValues(count: 100_000))
    let floatVector = VectorDenseBLAS(vectorFloatValues(count: 100_000))
    let doubleMatrixVector = VectorDenseBLAS(vectorValues(count: 384))
    let floatMatrixVector = VectorDenseBLAS(vectorFloatValues(count: 384))
    let doubleMatrixRows = matrixRows(rows: 384, columns: 384)
    let floatMatrixRows = matrixFloatRows(rows: 384, columns: 384)
    let doubleMatrix = MatrixDenseBLAS(doubleMatrixRows)
    let floatMatrix = MatrixDenseBLAS(floatMatrixRows)
    let doubleLeftRows = matrixRows(rows: 128, columns: 128)
    let floatLeftRows = matrixFloatRows(rows: 128, columns: 128)
    let doubleRightRows = matrixRows(rows: 128, columns: 128)
    let floatRightRows = matrixFloatRows(rows: 128, columns: 128)
    let doubleLeftMatrix = MatrixDenseBLAS(doubleLeftRows)
    let floatLeftMatrix = MatrixDenseBLAS(floatLeftRows)
    let doubleRightMatrix = MatrixDenseBLAS(doubleRightRows)
    let floatRightMatrix = MatrixDenseBLAS(floatRightRows)
    var doubleTensor = TensorDenseBLAS<Double>(shape: [16, 24, 16], initialValue: 0.0)
    var doubleRightTensor = TensorDenseBLAS<Double>(shape: [16, 16], initialValue: 0.0)
    var floatTensor = TensorDenseBLAS<Float>(shape: [16, 24, 16], initialValue: 0.0)
    var floatRightTensor = TensorDenseBLAS<Float>(shape: [16, 16], initialValue: 0.0)
    fillTensor(&doubleTensor)
    fillTensor(&doubleRightTensor)
    fillFloatTensor(&floatTensor)
    fillFloatTensor(&floatRightTensor)
    var checksum = 0.0
    checksum += compareFloatBenchmark("vector add", "100,000", iterations: 10, double: {
        let result = doubleVector + doubleVector
        return result[0] + result[result.size - 1]
    }, float: {
        let result = floatVector + floatVector
        return Double(result[0] + result[result.size - 1])
    })
    checksum += compareFloatBenchmark("magnitude", "100,000", iterations: 10, double: {
        doubleVector.magnitude()
    }, float: {
        Double(floatVector.magnitude())
    })
    checksum += compareFloatBenchmark("matrix-vector multiply", "384x384", double: {
        let result = doubleMatrix * doubleMatrixVector
        return result[0] + result[result.size - 1]
    }, float: {
        let result = floatMatrix * floatMatrixVector
        return Double(result[0] + result[result.size - 1])
    })
    checksum += compareFloatBenchmark("matrix-matrix multiply", "128x128", double: {
        let result = doubleLeftMatrix * doubleRightMatrix
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, float: {
        let result = floatLeftMatrix * floatRightMatrix
        return Double(result[0, 0] + result[result.rows - 1, result.columns - 1])
    })
    checksum += compareFloatBenchmark("tensor contraction", "16x24x16,16x16", double: {
        let result = multiply(doubleTensor, ["i", "j", "k"], doubleRightTensor, ["k", "l"])
        return result[[0, 0, 0]] + result[[15, 23, 15]]
    }, float: {
        let result = multiply(floatTensor, ["i", "j", "k"], floatRightTensor, ["k", "l"])
        return Double(result[[0, 0, 0]] + result[[15, 23, 15]])
    })
    return checksum
}

func benchmarkComplexFloatScalars() -> Double {
    print("ComplexFloat vs. ComplexDouble")
    let complexVector = VectorDenseBLAS(vectorComplexValues(count: 100_000))
    let complexFloatVector = VectorDenseBLAS(vectorComplexFloatValues(count: 100_000))
    let complexMatrixVector = VectorDenseBLAS(vectorComplexValues(count: 384))
    let complexFloatMatrixVector = VectorDenseBLAS(vectorComplexFloatValues(count: 384))
    let complexMatrix = MatrixDenseBLAS(matrixComplexRows(rows: 384, columns: 384))
    let complexFloatMatrix = MatrixDenseBLAS(matrixComplexFloatRows(rows: 384, columns: 384))
    let complexLeftMatrix = MatrixDenseBLAS(matrixComplexRows(rows: 128, columns: 128))
    let complexRightMatrix = MatrixDenseBLAS(matrixComplexRows(rows: 128, columns: 128))
    let complexFloatLeftMatrix = MatrixDenseBLAS(matrixComplexFloatRows(rows: 128, columns: 128))
    let complexFloatRightMatrix = MatrixDenseBLAS(matrixComplexFloatRows(rows: 128, columns: 128))
    var complexTensor = TensorDenseBLAS<ComplexDouble>(shape: [16, 24, 16], initialValue: .zero)
    var complexRightTensor = TensorDenseBLAS<ComplexDouble>(shape: [16, 16], initialValue: .zero)
    var complexFloatTensor = TensorDenseBLAS<ComplexFloat>(shape: [16, 24, 16], initialValue: .zero)
    var complexFloatRightTensor = TensorDenseBLAS<ComplexFloat>(shape: [16, 16], initialValue: .zero)
    fillComplexTensor(&complexTensor)
    fillComplexTensor(&complexRightTensor)
    fillComplexFloatTensor(&complexFloatTensor)
    fillComplexFloatTensor(&complexFloatRightTensor)
    var checksum = 0.0
    checksum += compareComplexFloatBenchmark("vector add", "100,000", iterations: 10, complex: {
        let result = complexVector + complexVector
        return result[0].real + result[result.size - 1].real
    }, complexFloat: {
        let result = complexFloatVector + complexFloatVector
        return Double(result[0].real + result[result.size - 1].real)
    })
    checksum += compareComplexFloatBenchmark("magnitude", "100,000", iterations: 10, complex: {
        complexVector.magnitude()
    }, complexFloat: {
        Double(complexFloatVector.magnitude())
    })
    checksum += compareComplexFloatBenchmark("matrix-vector multiply", "384x384", complex: {
        let result = complexMatrix * complexMatrixVector
        return result[0].real + result[result.size - 1].real
    }, complexFloat: {
        let result = complexFloatMatrix * complexFloatMatrixVector
        return Double(result[0].real + result[result.size - 1].real)
    })
    checksum += compareComplexFloatBenchmark("matrix-matrix multiply", "128x128", complex: {
        let result = complexLeftMatrix * complexRightMatrix
        return result[0, 0].real + result[result.rows - 1, result.columns - 1].real
    }, complexFloat: {
        let result = complexFloatLeftMatrix * complexFloatRightMatrix
        return Double(result[0, 0].real + result[result.rows - 1, result.columns - 1].real)
    })
    checksum += compareComplexFloatBenchmark("tensor contraction", "16x24x16,16x16", complex: {
        let result = multiply(complexTensor, ["i", "j", "k"], complexRightTensor, ["k", "l"])
        return result[[0, 0, 0]].real + result[[15, 23, 15]].real
    }, complexFloat: {
        let result = multiply(complexFloatTensor, ["i", "j", "k"], complexFloatRightTensor, ["k", "l"])
        return Double(result[[0, 0, 0]].real + result[[15, 23, 15]].real)
    })
    return checksum
}

func benchmarkAccelerateVsOpenBLAS() -> Double {
    print("Accelerate vs. OpenBLAS")
    #if canImport(Accelerate)
    let matrixVector = VectorDenseBLAS(vectorValues(count: 1_024))
    let largeMatrixVector = VectorDenseBLAS(vectorValues(count: 2_048))
    let hugeMatrixVector = VectorDenseBLAS(vectorValues(count: 4_096))
    let generatedRows = matrixRows(rows: 1_024, columns: 1_024)
    let largeGeneratedRows = matrixRows(rows: 2_048, columns: 2_048)
    let hugeGeneratedRows = matrixRows(rows: 4_096, columns: 4_096)
    let leftRows = matrixRows(rows: 256, columns: 256)
    let rightRows = matrixRows(rows: 256, columns: 256)
    let largeLeftRows = matrixRows(rows: 512, columns: 512)
    let largeRightRows = matrixRows(rows: 512, columns: 512)
    let hugeLeftRows = matrixRows(rows: 1_024, columns: 1_024)
    let hugeRightRows = matrixRows(rows: 1_024, columns: 1_024)
    let hugeFloatLeftRows = hugeLeftRows.map { $0.map(Float.init) }
    let hugeFloatRightRows = hugeRightRows.map { $0.map(Float.init) }
    let lapackRows = invertibleMatrixRows(size: 160)
    let largeLAPACKRows = invertibleMatrixRows(size: 256)
    let eigenRows = matrixRows(rows: 96, columns: 96)
    let largeEigenRows = matrixRows(rows: 160, columns: 160)
    var accelerateMatrix = MatrixDenseBLAS(generatedRows)
    var openBLASMatrix = MatrixDenseBLAS(generatedRows)
    var accelerateLargeMatrix = MatrixDenseBLAS(largeGeneratedRows)
    var openBLASLargeMatrix = MatrixDenseBLAS(largeGeneratedRows)
    var accelerateHugeMatrix = MatrixDenseBLAS(hugeGeneratedRows)
    var openBLASHugeMatrix = MatrixDenseBLAS(hugeGeneratedRows)
    var accelerateLeft = MatrixDenseBLAS(leftRows)
    var openBLASLeft = MatrixDenseBLAS(leftRows)
    var accelerateRight = MatrixDenseBLAS(rightRows)
    var openBLASRight = MatrixDenseBLAS(rightRows)
    var accelerateLargeLeft = MatrixDenseBLAS(largeLeftRows)
    var openBLASLargeLeft = MatrixDenseBLAS(largeLeftRows)
    var accelerateLargeRight = MatrixDenseBLAS(largeRightRows)
    var openBLASLargeRight = MatrixDenseBLAS(largeRightRows)
    var accelerateHugeLeft = MatrixDenseBLAS(hugeLeftRows)
    var openBLASHugeLeft = MatrixDenseBLAS(hugeLeftRows)
    var accelerateHugeRight = MatrixDenseBLAS(hugeRightRows)
    var openBLASHugeRight = MatrixDenseBLAS(hugeRightRows)
    var accelerateHugeFloatLeft = MatrixDenseBLAS<Float>(hugeFloatLeftRows)
    var openBLASHugeFloatLeft = MatrixDenseBLAS<Float>(hugeFloatLeftRows)
    var accelerateHugeFloatRight = MatrixDenseBLAS<Float>(hugeFloatRightRows)
    var openBLASHugeFloatRight = MatrixDenseBLAS<Float>(hugeFloatRightRows)
    var accelerateLAPACK = MatrixDenseBLAS(lapackRows)
    var openBLASLAPACK = MatrixDenseBLAS(lapackRows)
    var accelerateLargeLAPACK = MatrixDenseBLAS(largeLAPACKRows)
    var openBLASLargeLAPACK = MatrixDenseBLAS(largeLAPACKRows)
    var accelerateEigen = MatrixDenseBLAS(eigenRows)
    var openBLASEigen = MatrixDenseBLAS(eigenRows)
    var accelerateLargeEigen = MatrixDenseBLAS(largeEigenRows)
    var openBLASLargeEigen = MatrixDenseBLAS(largeEigenRows)
    accelerateMatrix.blasImplementation = .accelerate
    accelerateLargeMatrix.blasImplementation = .accelerate
    accelerateHugeMatrix.blasImplementation = .accelerate
    accelerateLeft.blasImplementation = .accelerate
    accelerateRight.blasImplementation = .accelerate
    accelerateLargeLeft.blasImplementation = .accelerate
    accelerateLargeRight.blasImplementation = .accelerate
    accelerateHugeLeft.blasImplementation = .accelerate
    accelerateHugeRight.blasImplementation = .accelerate
    accelerateHugeFloatLeft.blasImplementation = .accelerate
    accelerateHugeFloatRight.blasImplementation = .accelerate
    accelerateLAPACK.blasImplementation = .accelerate
    accelerateLargeLAPACK.blasImplementation = .accelerate
    accelerateEigen.blasImplementation = .accelerate
    accelerateLargeEigen.blasImplementation = .accelerate
    openBLASMatrix.blasImplementation = .openBLAS
    openBLASLargeMatrix.blasImplementation = .openBLAS
    openBLASHugeMatrix.blasImplementation = .openBLAS
    openBLASLeft.blasImplementation = .openBLAS
    openBLASRight.blasImplementation = .openBLAS
    openBLASLargeLeft.blasImplementation = .openBLAS
    openBLASLargeRight.blasImplementation = .openBLAS
    openBLASHugeLeft.blasImplementation = .openBLAS
    openBLASHugeRight.blasImplementation = .openBLAS
    openBLASHugeFloatLeft.blasImplementation = .openBLAS
    openBLASHugeFloatRight.blasImplementation = .openBLAS
    openBLASLAPACK.blasImplementation = .openBLAS
    openBLASLargeLAPACK.blasImplementation = .openBLAS
    openBLASEigen.blasImplementation = .openBLAS
    openBLASLargeEigen.blasImplementation = .openBLAS
    var checksum = 0.0
    checksum += compareImplementationBenchmark("matrix-vector multiply", "1,024x1,024", accelerate: {
        let result = accelerateMatrix * matrixVector
        return result[0] + result[result.size - 1]
    }, openBLAS: {
        let result = openBLASMatrix * matrixVector
        return result[0] + result[result.size - 1]
    })
    checksum += compareImplementationBenchmark("matrix-matrix multiply", "256x256", accelerate: {
        let result = accelerateLeft * accelerateRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, openBLAS: {
        let result = openBLASLeft * openBLASRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareImplementationBenchmark("matrix-vector multiply", "2,048x2,048", samples: 2, accelerate: {
        let result = accelerateLargeMatrix * largeMatrixVector
        return result[0] + result[result.size - 1]
    }, openBLAS: {
        let result = openBLASLargeMatrix * largeMatrixVector
        return result[0] + result[result.size - 1]
    })
    checksum += compareImplementationBenchmark("matrix-matrix multiply", "512x512", samples: 2, accelerate: {
        let result = accelerateLargeLeft * accelerateLargeRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, openBLAS: {
        let result = openBLASLargeLeft * openBLASLargeRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareImplementationBenchmark("matrix-vector multiply", "4,096x4,096", samples: 1, accelerate: {
        let result = accelerateHugeMatrix * hugeMatrixVector
        return result[0] + result[result.size - 1]
    }, openBLAS: {
        let result = openBLASHugeMatrix * hugeMatrixVector
        return result[0] + result[result.size - 1]
    })
    checksum += compareImplementationBenchmark("matrix-matrix multiply", "1,024x1,024", samples: 1, accelerate: {
        let result = accelerateHugeLeft * accelerateHugeRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, openBLAS: {
        let result = openBLASHugeLeft * openBLASHugeRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareImplementationBenchmark("Float matrix-matrix multiply", "1,024x1,024", samples: 1, accelerate: {
        let result = accelerateHugeFloatLeft * accelerateHugeFloatRight
        return Double(result[0, 0] + result[result.rows - 1, result.columns - 1])
    }, openBLAS: {
        let result = openBLASHugeFloatLeft * openBLASHugeFloatRight
        return Double(result[0, 0] + result[result.rows - 1, result.columns - 1])
    })
    checksum += compareImplementationBenchmark("determinant", "160x160", accelerate: {
        determinantChecksum(accelerateLAPACK.det)
    }, openBLAS: {
        determinantChecksum(openBLASLAPACK.det)
    })
    checksum += compareImplementationBenchmark("inverse", "160x160", accelerate: {
        let result = accelerateLAPACK.inverse()
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, openBLAS: {
        let result = openBLASLAPACK.inverse()
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareImplementationBenchmark("determinant", "256x256", samples: 2, accelerate: {
        determinantChecksum(accelerateLargeLAPACK.det)
    }, openBLAS: {
        determinantChecksum(openBLASLargeLAPACK.det)
    })
    checksum += compareImplementationBenchmark("inverse", "256x256", samples: 2, accelerate: {
        let result = accelerateLargeLAPACK.inverse()
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }, openBLAS: {
        let result = openBLASLargeLAPACK.inverse()
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    })
    checksum += compareImplementationBenchmark("eigen", "96x96", accelerate: {
        let result = accelerateEigen.eigen()
        return result.values[0].real + result.values[result.values.count - 1].real
    }, openBLAS: {
        let result = openBLASEigen.eigen()
        return result.values[0].real + result.values[result.values.count - 1].real
    })
    checksum += compareImplementationBenchmark("eigen", "160x160", samples: 2, accelerate: {
        let result = accelerateLargeEigen.eigen()
        return result.values[0].real + result.values[result.values.count - 1].real
    }, openBLAS: {
        let result = openBLASLargeEigen.eigen()
        return result.values[0].real + result.values[result.values.count - 1].real
    })
    return checksum
    #else
    print("  skipped: Accelerate is only available on Apple platforms")
    print("")
    return 0.0
    #endif
}

let swiftVersion = commandOutput("/usr/bin/env", ["swift", "--version"])
let platform = commandOutput("/usr/bin/env", ["uname", "-m"])

print("PlumeriaBenchmarks")
print("Swift: \(swiftVersion)")
print("Platform: \(platform)")
print("")
let blackHole = benchmarkVectors() + benchmarkMatrices() + benchmarkTensors()
    + benchmarkFloatScalars() + benchmarkComplexFloatScalars() + benchmarkAccelerateVsOpenBLAS()
writeBenchmarkResultToNullDevice(blackHole)
