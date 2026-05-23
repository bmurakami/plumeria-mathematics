import Foundation
import PlumeriaMathematics

#if canImport(Accelerate)
import Accelerate
#endif

func benchmarkMatrixAddMicro() -> Double {
    print("Dense Arithmetic Microbenchmark")
    var checksum = 0.0
    checksum += benchmarkVectorAddMicro(count: 100_000, samples: 5, iterations: 20)
    checksum += benchmarkMatrixAddMicro(rows: 256, columns: 256, samples: 5, iterations: 10)
    checksum += benchmarkMatrixAddMicro(rows: 1_024, columns: 1_024, samples: 5, iterations: 3)
    checksum += benchmarkTensorAddMicro(shape: [40, 40, 10], samples: 5, iterations: 20)
    return checksum
}

private func benchmarkVectorAddMicro(count: Int, samples: Int, iterations: Int) -> Double {
    let left = vectorValues(count: count)
    let right = left
    let leftVector = VectorDenseBLAS<Double>(left)
    let rightVector = leftVector
    print("  vector \(count)")
    let plain = measure(samples: samples, iterations: iterations) {
        let result = plainSwiftAdd(left, right)
        return result[0] + result[result.count - 1]
    }
    let vector = measure(samples: samples, iterations: iterations) {
        let result = leftVector + rightVector
        return result[0] + result[result.size - 1]
    }
    printMicroResult("plain Swift loop", plain.0)
    printMicroResult("VectorDenseBLAS +", vector.0)
    print("")
    return plain.1 + vector.1
}

private func benchmarkMatrixAddMicro(rows: Int, columns: Int, samples: Int, iterations: Int) -> Double {
    let left = columnMajorMatrixValues(rows: rows, columns: columns)
    let right = left
    let matrixRows = matrixRows(rows: rows, columns: columns)
    let leftMatrix = MatrixDenseBLAS<Double>(matrixRows)
    let rightMatrix = leftMatrix
    print("  \(rows)x\(columns)")
    let plain = measure(samples: samples, iterations: iterations) {
        let result = plainSwiftAdd(left, right)
        return result[0] + result[result.count - 1]
    }
    let unsafe = measure(samples: samples, iterations: iterations) {
        let result = unsafeBufferAdd(left, right)
        return result[0] + result[result.count - 1]
    }
    let axpy = measure(samples: samples, iterations: iterations) {
        let result = directAccelerateAxpy(left, right)
        return result[0] + result[result.count - 1]
    }
    let genericAxpy = measure(samples: samples, iterations: iterations) {
        let result = genericCastAxpy(left, right)
        return result[0] + result[result.count - 1]
    }
    let constructMatrix = measure(samples: samples, iterations: iterations) {
        let values = directAccelerateAxpy(left, right)
        let result = MatrixDenseBLAS<Double>(rows: rows, columns: columns, values: values)
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }
    let matrix = measure(samples: samples, iterations: iterations) {
        let result = leftMatrix + rightMatrix
        return result[0, 0] + result[result.rows - 1, result.columns - 1]
    }
    printMicroResult("plain Swift loop", plain.0)
    printMicroResult("unsafe-buffer loop", unsafe.0)
    printMicroResult("direct Accelerate axpy", axpy.0)
    printMicroResult("generic cast axpy", genericAxpy.0)
    printMicroResult("construct matrix from axpy", constructMatrix.0)
    printMicroResult("MatrixDenseBLAS +", matrix.0)
    print("")
    return plain.1 + unsafe.1 + axpy.1 + genericAxpy.1 + constructMatrix.1 + matrix.1
}

private func benchmarkTensorAddMicro(shape: [Int], samples: Int, iterations: Int) -> Double {
    let count = shape.reduce(1, *)
    let left = vectorValues(count: count)
    let right = left
    let leftTensor = TensorDenseBLAS<Double>(shape: shape, elements: left)
    let rightTensor = leftTensor
    print("  tensor \(shape.map(String.init).joined(separator: "x"))")
    let plain = measure(samples: samples, iterations: iterations) {
        let result = plainSwiftAdd(left, right)
        return result[0] + result[result.count - 1]
    }
    let tensor = measure(samples: samples, iterations: iterations) {
        let result = leftTensor + rightTensor
        return result[[0, 0, 0]] + result[[shape[0] - 1, shape[1] - 1, shape[2] - 1]]
    }
    printMicroResult("plain Swift loop", plain.0)
    printMicroResult("TensorDenseBLAS +", tensor.0)
    print("")
    return plain.1 + tensor.1
}

private func printMicroResult(_ label: String, _ result: TimedResult) {
    print("    \(label): median \(format(result.median)) ms, best \(format(result.best)) ms")
}

@inline(never)
private func plainSwiftAdd(_ left: [Double], _ right: [Double]) -> [Double] {
    var result = Array(repeating: 0.0, count: left.count)
    for index in 0..<left.count { result[index] = left[index] + right[index] }
    return result
}

@inline(never)
private func unsafeBufferAdd(_ left: [Double], _ right: [Double]) -> [Double] {
    var result = Array(repeating: 0.0, count: left.count)
    left.withUnsafeBufferPointer { left in
        right.withUnsafeBufferPointer { right in
            result.withUnsafeMutableBufferPointer { result in
                for index in 0..<left.count { result[index] = left[index] + right[index] }
            }
        }
    }
    return result
}

@inline(never)
private func directAccelerateAxpy(_ left: [Double], _ right: [Double]) -> [Double] {
    var result = left
    #if canImport(Accelerate)
    let n = Int32(left.count)
    let alpha = 1.0
    let inc = Int32(1)
    right.withUnsafeBufferPointer { right in
        result.withUnsafeMutableBufferPointer { result in
            Accelerate.cblas_daxpy(n, alpha, right.baseAddress!, inc, result.baseAddress!, inc)
        }
    }
    #else
    for index in 0..<right.count { result[index] += right[index] }
    #endif
    return result
}

@inline(never)
private func genericCastAxpy<S: PluScalar>(_ left: [S], _ right: [S]) -> [S] {
    switch S.self {
    case is Double.Type:
        let x = right as! [Double]
        var y = left as! [Double]
        #if canImport(Accelerate)
        let n = Int32(y.count)
        let alpha = 1.0
        let inc = Int32(1)
        x.withUnsafeBufferPointer { x in
            y.withUnsafeMutableBufferPointer { y in
                Accelerate.cblas_daxpy(n, alpha, x.baseAddress!, inc, y.baseAddress!, inc)
            }
        }
        #else
        for index in 0..<x.count { y[index] += x[index] }
        #endif
        return y as! [S]
    default:
        fatalError("Unsupported scalar type")
    }
}

private func columnMajorMatrixValues(rows: Int, columns: Int) -> [Double] {
    var values: [Double] = []
    values.reserveCapacity(rows * columns)
    for column in 0..<columns {
        for row in 0..<rows {
            values.append(Double(((row * 31 + column * 17) % 101) - 50) / 11.0)
        }
    }
    return values
}
