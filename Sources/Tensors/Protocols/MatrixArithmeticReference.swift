import Foundation

public protocol MatrixArithmeticReference: MatrixArithmetic where Self: PluMatrix {}

extension MatrixArithmeticReference {
    public var det: S {
        precondition(rows == columns, "Determinant requires a square matrix")
        var values = toArray()
        var sign: S = 1
        var result: S = 1
        for column in 0..<columns {
            guard let pivot = (column..<rows).first(where: { values[$0][column] != .zero }) else { return .zero }
            if pivot != column {
                values.swapAt(pivot, column)
                sign = -sign
            }
            let pivotValue = values[column][column]
            result *= pivotValue
            if column + 1 < rows {
                for row in (column + 1)..<rows {
                    let factor = values[row][column] / pivotValue
                    for entry in column..<columns {
                        values[row][entry] -= factor * values[column][entry]
                    }
                }
            }
        }
        return sign * result
    }

    public func inverse() -> Self {
        precondition(rows == columns, "Inverse requires a square matrix")
        var left = toArray()
        var right = Self.identityArray(size: rows)
        for column in 0..<columns {
            guard let pivot = (column..<rows).first(where: { left[$0][column] != .zero }) else {
                preconditionFailure("Matrix must be invertible")
            }
            if pivot != column {
                left.swapAt(pivot, column)
                right.swapAt(pivot, column)
            }
            let pivotValue = left[column][column]
            for entry in 0..<columns {
                left[column][entry] = left[column][entry] / pivotValue
                right[column][entry] = right[column][entry] / pivotValue
            }
            for row in 0..<rows where row != column {
                let factor = left[row][column]
                for entry in 0..<columns {
                    left[row][entry] -= factor * left[column][entry]
                    right[row][entry] -= factor * right[column][entry]
                }
            }
        }
        return Self(right)
    }

    private static func identityArray(size: Int) -> [[S]] {
        (0..<size).map { row in
            (0..<size).map { column in row == column ? 1 : 0 }
        }
    }
}

extension MatrixArithmeticReference where Self: MatrixEigen, S == Double {
    public func eigen() -> Eigen {
        precondition(rows == columns, "Eigen decomposition requires a square matrix")
        if rows == 1 {
            return Eigen(values: [Complex(self[0, 0], 0.0)], vectors: MatrixDenseBLAS<Complex>([[Complex(1.0, 0.0)]]))
        }
        precondition(rows == 2, "Reference eigen decomposition currently supports 1x1 and 2x2 matrices")
        let a = Complex(self[0, 0], 0.0), b = Complex(self[0, 1], 0.0)
        let c = Complex(self[1, 0], 0.0), d = Complex(self[1, 1], 0.0)
        let trace = a + d
        let determinant = a * d - b * c
        let root = squareRoot(trace * trace - 4.0 * determinant)
        let values = [(trace + root) / 2.0, (trace - root) / 2.0]
        let first = eigenvector(a: a, b: b, c: c, d: d, value: values[0], fallbackColumn: 0)
        let second = eigenvector(a: a, b: b, c: c, d: d, value: values[1], fallbackColumn: 1)
        return Eigen(values: values, vectors: MatrixDenseBLAS<Complex>(rows: 2, columns: 2, values: [
            first[0], first[1], second[0], second[1]
        ]))
    }

    private func eigenvector(
        a: Complex, b: Complex, c: Complex, d: Complex, value: Complex, fallbackColumn: Int
    ) -> [Complex] {
        let first = [b, value - a]
        if !isZero(first) { return first }
        let second = [value - d, c]
        if !isZero(second) { return second }
        return fallbackColumn == 0 ? [Complex(1.0, 0.0), Complex(0.0, 0.0)] : [Complex(0.0, 0.0), Complex(1.0, 0.0)]
    }

    private func isZero(_ vector: [Complex]) -> Bool {
        vector.allSatisfy { $0.length <= 1e-12 }
    }

    private func squareRoot(_ value: Complex) -> Complex {
        if value.imaginary == 0.0 && value.real >= 0.0 { return Complex(Foundation.sqrt(value.real), 0.0) }
        let length = value.length
        let real = Foundation.sqrt((length + value.real) / 2.0)
        let imaginarySign = value.imaginary < 0.0 ? -1.0 : 1.0
        let imaginary = imaginarySign * Foundation.sqrt((length - value.real) / 2.0)
        return Complex(real, imaginary)
    }
}
