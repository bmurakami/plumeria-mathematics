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
