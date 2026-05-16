import Foundation
import PlumeriaMathematics

struct TimedResult {
    let best: Double
    let median: Double
}

func runBenchmark(
    _ operation: String,
    _ implementation: String,
    _ size: String,
    samples: Int = 5,
    iterations: Int = 1,
    operation body: () -> Double
) -> Double {
    var total = body()
    var times: [Double] = []
    for _ in 0..<samples {
        let start = DispatchTime.now().uptimeNanoseconds
        var checksum = 0.0
        for _ in 0..<iterations { checksum += body() }
        let end = DispatchTime.now().uptimeNanoseconds
        total += checksum
        times.append(Double(end - start) / Double(iterations) / 1_000_000.0)
    }
    let result = summarize(times)
    print("\(operation),\(implementation),\(size),\(samples),\(format(result.best)),\(format(result.median))")
    return total
}

func summarize(_ values: [Double]) -> TimedResult {
    let sorted = values.sorted()
    return TimedResult(best: sorted[0], median: sorted[sorted.count / 2])
}

func format(_ value: Double) -> String {
    String(format: "%.4f", value)
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
    let small = vectorValues(count: 10_000)
    let large = vectorValues(count: 100_000)
    let referenceSmall = VectorDenseReference(small)
    let referenceLarge = VectorDenseReference(large)
    let blasSmall = VectorDenseBLAS(small)
    let blasLarge = VectorDenseBLAS(large)
    var checksum = 0.0
    checksum += runBenchmark("vector add", "reference", "10000", iterations: 20) {
        let result = referenceSmall + referenceSmall
        return result[0] + result[result.size - 1]
    }
    checksum += runBenchmark("vector add", "blas", "10000", iterations: 20) {
        let result = blasSmall + blasSmall
        return result[0] + result[result.size - 1]
    }
    checksum += runBenchmark("vector scale", "reference", "100000", iterations: 10) {
        let result = referenceLarge * 1.25
        return result[0] + result[result.size - 1]
    }
    checksum += runBenchmark("vector scale", "blas", "100000", iterations: 10) {
        let result = blasLarge * 1.25
        return result[0] + result[result.size - 1]
    }
    checksum += runBenchmark("vector magnitude", "reference", "100000", iterations: 10) {
        referenceLarge.magnitude()
    }
    checksum += runBenchmark("vector magnitude", "blas", "100000", iterations: 10) {
        blasLarge.magnitude()
    }
    return checksum
}

func benchmarkMatrices() -> Double {
    let addRows = matrixRows(rows: 256, columns: 256)
    let vector = VectorDenseBLAS(vectorValues(count: 384))
    let mvRows = matrixRows(rows: 384, columns: 384)
    let mmLeftRows = matrixRows(rows: 128, columns: 128)
    let mmRightRows = matrixRows(rows: 128, columns: 128)
    let referenceAdd = MatrixDenseReference(addRows)
    let blasAdd = MatrixDenseBLAS(addRows)
    let referenceMV = MatrixDenseReference(mvRows)
    let blasMV = MatrixDenseBLAS(mvRows)
    let referenceMMLeft = MatrixDenseReference(mmLeftRows)
    let referenceMMRight = MatrixDenseReference(mmRightRows)
    let blasMMLeft = MatrixDenseBLAS(mmLeftRows)
    let blasMMRight = MatrixDenseBLAS(mmRightRows)
    var checksum = 0.0
    checksum += runBenchmark("matrix add", "reference", "256x256", iterations: 5) {
        let result = referenceAdd + referenceAdd
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }
    checksum += runBenchmark("matrix add", "blas", "256x256", iterations: 5) {
        let result = blasAdd + blasAdd
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }
    checksum += runBenchmark("matrix-vector multiply", "reference", "384x384") {
        let result = referenceMV * vector
        return result[0] + result[result.size - 1]
    }
    checksum += runBenchmark("matrix-vector multiply", "blas", "384x384") {
        let result = blasMV * vector
        return result[0] + result[result.size - 1]
    }
    checksum += runBenchmark("matrix-matrix multiply", "reference", "128x128") {
        let result = referenceMMLeft * referenceMMRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }
    checksum += runBenchmark("matrix-matrix multiply", "blas", "128x128") {
        let result = blasMMLeft * blasMMRight
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }
    return checksum
}

func benchmarkTensors() -> Double {
    var referenceAdd = TensorDenseReference<Double>(shape: [40, 40, 10], initialValue: 0.0)
    var blasAdd = TensorDenseBLAS<Double>(shape: [40, 40, 10], initialValue: 0.0)
    var referenceLeft = TensorDenseReference<Double>(shape: [16, 24, 16], initialValue: 0.0)
    var referenceRight = TensorDenseReference<Double>(shape: [16, 16], initialValue: 0.0)
    var blasLeft = TensorDenseBLAS<Double>(shape: [16, 24, 16], initialValue: 0.0)
    var blasRight = TensorDenseBLAS<Double>(shape: [16, 16], initialValue: 0.0)
    fillTensor(&referenceAdd)
    fillTensor(&blasAdd)
    fillTensor(&referenceLeft)
    fillTensor(&referenceRight)
    fillTensor(&blasLeft)
    fillTensor(&blasRight)
    var checksum = 0.0
    checksum += runBenchmark("tensor add", "reference", "40x40x10", iterations: 3) {
        let result = referenceAdd + referenceAdd
        return result[[0, 0, 0]] + result[[39, 39, 9]]
    }
    checksum += runBenchmark("tensor add", "blas", "40x40x10", iterations: 3) {
        let result = blasAdd + blasAdd
        return result[[0, 0, 0]] + result[[39, 39, 9]]
    }
    checksum += runBenchmark("tensor contraction", "reference", "16x24x16,16x16") {
        let result = multiply(referenceLeft, ["i", "j", "k"], referenceRight, ["k", "l"])
        return result[[0, 0, 0]] + result[[15, 23, 15]]
    }
    checksum += runBenchmark("tensor contraction", "blas", "16x24x16,16x16") {
        let result = multiply(blasLeft, ["i", "j", "k"], blasRight, ["k", "l"])
        return result[[0, 0, 0]] + result[[15, 23, 15]]
    }
    return checksum
}

let swiftVersion = commandOutput("/usr/bin/env", ["swift", "--version"])
let platform = commandOutput("/usr/bin/env", ["uname", "-m"])

print("PlumeriaBenchmarks")
print("swift,\(swiftVersion)")
print("platform,\(platform)")
print("operation,implementation,size,samples,best_ms,median_ms")
let blackHole = benchmarkVectors() + benchmarkMatrices() + benchmarkTensors()
print("blackHole,\(blackHole)")
