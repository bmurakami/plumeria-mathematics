import Foundation
import PlumeriaMathematics

private let waveNX = 300
private let waveNY = 300

final class PlumeriaWaveBenchmark {
    private var u_n = MatrixDenseBLAS<Double>(rows: waveNY, columns: waveNX)
    private var u_nm1 = MatrixDenseBLAS<Double>(rows: waveNY, columns: waveNX)
    private var u_nm2 = MatrixDenseBLAS<Double>(rows: waveNY, columns: waveNX)
    private var pixelData = Array(repeating: UInt8(255), count: waveNX * waveNY * 3)
    private let alpha = 0.25

    init() {
        for dx in -2..<2 {
            for dy in -2..<2 { u_n[waveNY / 2 + dy, waveNX / 2 + dx] = 120.0 }
        }
    }

    func step() -> Double {
        swap(&u_nm2, &u_nm1)
        swap(&u_nm1, &u_n)
        let r = 1..<(waveNY - 1)
        let c = 1..<(waveNX - 1)
        let u_center: MatrixDenseBLAS<Double> = u_nm1[r, c]
        let u_xx: MatrixDenseBLAS<Double> = u_nm1[r, 0..<(waveNX - 2)] + u_nm1[r, 2..<waveNX]
        let u_yy: MatrixDenseBLAS<Double> = u_nm1[0..<(waveNY - 2), c] + u_nm1[2..<waveNY, c]
        let laplacian: MatrixDenseBLAS<Double> = u_xx + u_yy - 4 * u_center
        let u_next: MatrixDenseBLAS<Double> = (alpha * laplacian + 2 * u_center - u_nm2[r, c]) * 0.995
        u_n[r, c] = u_next
        return u_n[1, 1] + u_n[waveNY - 2, waveNX - 2]
    }

    func renderPixels() -> Double {
        pixelData.withUnsafeMutableBufferPointer { pixels in
            for y in 0..<waveNY {
                let row = y * waveNX
                for x in 0..<waveNX {
                    let offset = (row + x) * 3
                    pixels[offset] = clipped(u_n[y, x] + 128)
                    pixels[offset + 1] = clipped(u_nm1[y, x] + 128)
                    pixels[offset + 2] = clipped(u_nm2[y, x] + 128)
                }
            }
        }
        return Double(pixelData[0]) + Double(pixelData[pixelData.count - 1])
    }

    private func clipped(_ value: Double) -> UInt8 {
        UInt8(max(0.0, min(255.0, value.rounded())))
    }
}

final class RawSwiftWaveBenchmark {
    private var u_n = Array(repeating: 0.0, count: waveNX * waveNY)
    private var u_nm1 = Array(repeating: 0.0, count: waveNX * waveNY)
    private var u_nm2 = Array(repeating: 0.0, count: waveNX * waveNY)
    private var pixelData = Array(repeating: UInt8(255), count: waveNX * waveNY * 3)
    private let alpha = 0.25

    init() {
        for dx in -2..<2 {
            for dy in -2..<2 { u_n[index(x: waveNX / 2 + dx, y: waveNY / 2 + dy)] = 120.0 }
        }
    }

    func step() -> Double {
        swap(&u_nm2, &u_nm1)
        swap(&u_nm1, &u_n)
        for y in 1..<(waveNY - 1) {
            let row = y * waveNX
            let up = (y - 1) * waveNX
            let down = (y + 1) * waveNX
            for x in 1..<(waveNX - 1) {
                let center = row + x
                let laplacian = u_nm1[center - 1] + u_nm1[center + 1]
                    + u_nm1[up + x] + u_nm1[down + x] - 4 * u_nm1[center]
                u_n[center] = (alpha * laplacian + 2 * u_nm1[center] - u_nm2[center]) * 0.995
            }
        }
        return u_n[index(x: 1, y: 1)] + u_n[index(x: waveNX - 2, y: waveNY - 2)]
    }

    func renderPixels() -> Double {
        for y in 0..<waveNY {
            let row = y * waveNX
            for x in 0..<waveNX {
                let cell = row + x
                let offset = cell * 3
                pixelData[offset] = clipped(u_n[cell] + 128)
                pixelData[offset + 1] = clipped(u_nm1[cell] + 128)
                pixelData[offset + 2] = clipped(u_nm2[cell] + 128)
            }
        }
        return Double(pixelData[0]) + Double(pixelData[pixelData.count - 1])
    }

    private func index(x: Int, y: Int) -> Int { y * waveNX + x }

    private func clipped(_ value: Double) -> UInt8 {
        UInt8(max(0.0, min(255.0, value.rounded())))
    }
}

final class RawSwiftColumnMajorWaveBenchmark {
    private var u_n = Array(repeating: 0.0, count: waveNX * waveNY)
    private var u_nm1 = Array(repeating: 0.0, count: waveNX * waveNY)
    private var u_nm2 = Array(repeating: 0.0, count: waveNX * waveNY)
    private let alpha = 0.25

    init() {
        for dx in -2..<2 {
            for dy in -2..<2 { u_n[index(row: waveNY / 2 + dy, column: waveNX / 2 + dx)] = 120.0 }
        }
    }

    func step() -> Double {
        swap(&u_nm2, &u_nm1)
        swap(&u_nm1, &u_n)
        for column in 1..<(waveNX - 1) {
            let left = (column - 1) * waveNY
            let centerColumn = column * waveNY
            let right = (column + 1) * waveNY
            for row in 1..<(waveNY - 1) {
                let center = centerColumn + row
                let laplacian = u_nm1[left + row] + u_nm1[right + row]
                    + u_nm1[center - 1] + u_nm1[center + 1] - 4 * u_nm1[center]
                u_n[center] = (alpha * laplacian + 2 * u_nm1[center] - u_nm2[center]) * 0.995
            }
        }
        return u_n[index(row: 1, column: 1)] + u_n[index(row: waveNY - 2, column: waveNX - 2)]
    }

    private func index(row: Int, column: Int) -> Int { row + waveNY * column }
}

func benchmarkWaveStencil() -> Double {
    print("Wave stencil")
    print("shape: \(waveNX)x\(waveNY)")
    print("")
    var checksum = 0.0
    let samples = 5
    let iterations = 100
    let plumeriaUpdate = PlumeriaWaveBenchmark()
    checksum += printWaveResult("PluMath repeated update", measure(samples: samples, iterations: iterations) {
        plumeriaUpdate.step()
    })
    let plumeriaRender = PlumeriaWaveBenchmark()
    checksum += printWaveResult("PluMath pixel fill", measure(samples: samples, iterations: iterations) {
        plumeriaRender.renderPixels()
    })
    let rawUpdate = RawSwiftWaveBenchmark()
    checksum += printWaveResult("raw Swift repeated update", measure(samples: samples, iterations: iterations) {
        rawUpdate.step()
    })
    let rawColumnMajorUpdate = RawSwiftColumnMajorWaveBenchmark()
    checksum += printWaveResult("raw Swift column-major update", measure(samples: samples, iterations: iterations) {
        rawColumnMajorUpdate.step()
    })
    let rawRender = RawSwiftWaveBenchmark()
    checksum += printWaveResult("raw Swift pixel fill", measure(samples: samples, iterations: iterations) {
        rawRender.renderPixels()
    })
    checksum += benchmarkMatrixSlicePrimitives()
    return checksum
}

private func benchmarkMatrixSlicePrimitives() -> Double {
    print("Matrix slice primitives")
    print("")
    let samples = 5
    let iterations = 100
    let matrix = MatrixDenseBLAS<Double>(
        rows: waveNY, columns: waveNX, values: (0..<(waveNX * waveNY)).map { Double($0 % 17) }
    )
    let r = 1..<(waveNY - 1)
    let c = 1..<(waveNX - 1)
    var destination = matrix
    var checksum = 0.0
    checksum += printWaveResult("slice view", measure(samples: samples, iterations: iterations) {
        let slice: MatrixDenseBLAS<Double> = matrix[r, c]
        return slice[0, 0]
    })
    checksum += printWaveResult("slice add", measure(samples: samples, iterations: iterations) {
        let sum: MatrixDenseBLAS<Double> = matrix[r, 0..<(waveNX - 2)] + matrix[r, 2..<waveNX]
        return sum[0, 0]
    })
    checksum += printWaveResult("slice scalar multiply", measure(samples: samples, iterations: iterations) {
        let scaled: MatrixDenseBLAS<Double> = 4 * matrix[r, c]
        return scaled[0, 0]
    })
    checksum += printWaveResult("slice assign", measure(samples: samples, iterations: iterations) {
        let slice: MatrixDenseBLAS<Double> = matrix[r, c]
        destination[r, c] = slice
        return destination[1, 1]
    })
    checksum += benchmarkWaveExpressionPrimitives(matrix: matrix, rows: r, columns: c, samples: samples,
                                                  iterations: iterations)
    return checksum
}

private func benchmarkWaveExpressionPrimitives(
    matrix: MatrixDenseBLAS<Double>, rows r: Range<Int>, columns c: Range<Int>, samples: Int, iterations: Int
) -> Double {
    print("Wave expression primitives")
    print("")
    var destination = matrix
    let expression = waveExpression(matrix: matrix, rows: r, columns: c)
    var checksum = 0.0
    checksum += printWaveResult("wave expression build", measure(samples: samples, iterations: iterations) {
        let result = waveExpression(matrix: matrix, rows: r, columns: c)
        return result[0, 0]
    })
    checksum += printWaveResult("wave expression assign", measure(samples: samples, iterations: iterations) {
        destination[r, c] = expression
        return destination[1, 1]
    })
    return checksum
}

private func waveExpression(
    matrix: MatrixDenseBLAS<Double>, rows r: Range<Int>, columns c: Range<Int>
) -> MatrixDenseBLAS<Double> {
    let u_center: MatrixDenseBLAS<Double> = matrix[r, c]
    let u_xx: MatrixDenseBLAS<Double> = matrix[r, 0..<(waveNX - 2)] + matrix[r, 2..<waveNX]
    let u_yy: MatrixDenseBLAS<Double> = matrix[0..<(waveNY - 2), c] + matrix[2..<waveNY, c]
    let laplacian: MatrixDenseBLAS<Double> = u_xx + u_yy - 4 * u_center
    return (0.25 * laplacian + 2 * u_center - matrix[r, c]) * 0.995
}

private func printWaveResult(_ operation: String, _ result: (TimedResult, Double)) -> Double {
    print("  \(operation)")
    print("    median \(format(result.0.median)) ms, best \(format(result.0.best)) ms")
    print("    checksum \(format(result.1))")
    print("")
    return result.1
}
