import Foundation

public protocol MatrixArithmeticReference: MatrixArithmetic where Self: PluMatrix {}

extension MatrixArithmeticReference {
    public var det: S {
        precondition(rows == columns, "Determinant requires a square matrix")
        var values = toArray()
        var sign: S = 1
        var result: S = 1
        for j in 0..<columns {
            guard let pivot = (j..<rows).first(where: { values[$0][j] != .zero }) else { return .zero }
            if pivot != j {
                values.swapAt(pivot, j)
                sign = -sign
            }
            let pivotValue = values[j][j]
            result *= pivotValue
            if j + 1 < rows {
                for i in (j + 1)..<rows {
                    let factor = values[i][j] / pivotValue
                    for k in j..<columns {
                        values[i][k] -= factor * values[j][k]
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
        for j in 0..<columns {
            guard let pivot = (j..<rows).first(where: { left[$0][j] != .zero }) else {
                preconditionFailure("Matrix must be invertible")
            }
            if pivot != j {
                left.swapAt(pivot, j)
                right.swapAt(pivot, j)
            }
            let pivotValue = left[j][j]
            for k in 0..<columns {
                left[j][k] = left[j][k] / pivotValue
                right[j][k] = right[j][k] / pivotValue
            }
            for i in 0..<rows where i != j {
                let factor = left[i][j]
                for k in 0..<columns {
                    left[i][k] -= factor * left[j][k]
                    right[i][k] -= factor * right[j][k]
                }
            }
        }
        return Self(right)
    }

    private static func identityArray(size: Int) -> [[S]] {
        (0..<size).map { i in
            (0..<size).map { j in i == j ? 1 : 0 }
        }
    }
}

extension MatrixArithmeticReference where Self: MatrixEigen, S == Double {
    public func eigen() -> Eigen<Eigenvalue, Eigenvectors> where Eigenvalue == ComplexDouble {
        precondition(rows == columns, "Eigen decomposition requires a square matrix")
        if rows == 1 {
            return Eigen(values: [ComplexDouble(self[0, 0], 0.0)], vectors: Eigenvectors([[ComplexDouble(1.0, 0.0)]]))
        }
        precondition(rows == 2, "Reference eigen decomposition currently supports 1x1 and 2x2 matrices")
        let a = ComplexDouble(self[0, 0], 0.0), b = ComplexDouble(self[0, 1], 0.0)
        let c = ComplexDouble(self[1, 0], 0.0), d = ComplexDouble(self[1, 1], 0.0)
        let trace = a + d
        let determinant = a * d - b * c
        let root = squareRoot(trace * trace - 4.0 * determinant)
        let values = [(trace + root) / 2.0, (trace - root) / 2.0]
        let first = eigenvector(a: a, b: b, c: c, d: d, value: values[0], fallbackColumn: 0)
        let second = eigenvector(a: a, b: b, c: c, d: d, value: values[1], fallbackColumn: 1)
        return Eigen(values: values, vectors: Eigenvectors([[first[0], second[0]], [first[1], second[1]]]))
    }

    private func eigenvector(
        a: ComplexDouble, b: ComplexDouble, c: ComplexDouble, d: ComplexDouble,
        value: ComplexDouble, fallbackColumn: Int
    ) -> [ComplexDouble] {
        let first = [b, value - a]
        if !isZero(first) { return first }
        let second = [value - d, c]
        if !isZero(second) { return second }
        if fallbackColumn == 0 { return [ComplexDouble(1.0, 0.0), ComplexDouble(0.0, 0.0)] }
        return [ComplexDouble(0.0, 0.0), ComplexDouble(1.0, 0.0)]
    }

    private func isZero(_ vector: [ComplexDouble]) -> Bool {
        vector.allSatisfy { $0.length <= 1e-12 }
    }

    private func squareRoot(_ value: ComplexDouble) -> ComplexDouble {
        if value.imaginary == 0.0 && value.real >= 0.0 { return ComplexDouble(Foundation.sqrt(value.real), 0.0) }
        let length = value.length
        let real = Foundation.sqrt((length + value.real) / 2.0)
        let imaginarySign = value.imaginary < 0.0 ? -1.0 : 1.0
        let imaginary = imaginarySign * Foundation.sqrt((length - value.real) / 2.0)
        return ComplexDouble(real, imaginary)
    }
}
