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

func ratio(reference: Double, blas: Double) -> String {
    if reference == 0 && blas == 0 { return "same speed" }
    if reference == 0 { return "reference faster" }
    if blas == 0 { return "BLAS faster" }
    if reference < blas {
        return "reference \(format(blas / reference))x faster"
    }
    return "BLAS \(format(reference / blas))x faster"
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

func vectorValues(count: Int) -> [Double] {
    (0..<count).map { Double(($0 % 97) - 48) / 7.0 }
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

func fillTensor<T: TensorMultiplication>(_ tensor: inout T) where T.S == Double {
    for index in indexCombinations(for: tensor.shape) {
        let weighted = index.enumerated().reduce(0) { $0 + ($1.offset + 1) * ($1.element + 1) }
        tensor[index] = Double((weighted % 29) - 14) / 5.0
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

let swiftVersion = commandOutput("/usr/bin/env", ["swift", "--version"])
let platform = commandOutput("/usr/bin/env", ["uname", "-m"])

print("PlumeriaBenchmarks")
print("Swift: \(swiftVersion)")
print("Platform: \(platform)")
print("")
let blackHole = benchmarkVectors() + benchmarkMatrices() + benchmarkTensors()
print("Checksum: \(blackHole)")
