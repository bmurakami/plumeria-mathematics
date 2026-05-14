#if canImport(Accelerate)
import AccelerateWrapper
#else
import OpenBLASWrapper
#endif

public protocol PluMatrix: PluTensor, TensorStructure {
    associatedtype S: PluScalar
    
    var rows: Int { get }
    var columns: Int { get }
    subscript(i: Int, j: Int) -> S { get set }
    
    init(rows: Int, columns: Int, initialValue: S)
    init(_ values: [[S]])

    func times<V: PluVector>(_ v: V) -> V where V.S == S
    func times<M: PluMatrix>(_ m: M) -> Self where M.S == S
    func transpose() -> Self
    func toArray(round: Bool) -> [[S]]
    func flatten(columnMajorOrder: Bool) -> [S]
}

public struct Eigen: Equatable {
    public let values: [Complex]
    public let vectors: MatrixDenseBLAS<Complex>

    public init(values: [Complex], vectors: MatrixDenseBLAS<Complex>) {
        self.values = values
        self.vectors = vectors
    }
}

extension PluMatrix {
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { 2 }
    public var t: Self { transpose() }
    public var tr: S {
        precondition(rows == columns, "Trace requires a square matrix")
        var sum = S.zero
        for index in 0..<rows {
            sum += self[index, index]
        }
        return sum
    }
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

    public func toArray() -> [[S]] { return toArray(round: false) }
    public func flatten() -> [S] { return flatten(columnMajorOrder: true) }

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

    public static func identity(size: Int) -> Self {
        precondition(size > 0, "Identity matrix size must be positive")
        return Self(identityArray(size: size))
    }

    private static func identityArray(size: Int) -> [[S]] {
        (0..<size).map { row in
            (0..<size).map { column in row == column ? 1 : 0 }
        }
    }
}

extension PluMatrix where S == Double {
    public func eigen() -> Eigen {
        precondition(rows == columns, "Eigen decomposition requires a square matrix")
        let n = Int32(rows)
        var matrix = flatten()
        var real = Array(repeating: 0.0, count: rows)
        var imaginary = Array(repeating: 0.0, count: rows)
        var vectors = Array(repeating: 0.0, count: rows * columns)
        #if canImport(Accelerate)
        let info = AccelerateOperations.dgeev(n, &matrix, &real, &imaginary, &vectors)
        #else
        let info = OpenBLASOperations.dgeev(n, &matrix, &real, &imaginary, &vectors)
        #endif
        precondition(info == 0, "Eigen decomposition failed with LAPACK info \(info)")
        return Eigen(values: eigenvalues(real: real, imaginary: imaginary),
                     vectors: eigenvectors(real: real, imaginary: imaginary, vectors: vectors))
    }

    private func eigenvalues(real: [Double], imaginary: [Double]) -> [Complex] {
        zip(real, imaginary).map { Complex($0, $1) }
    }

    private func eigenvectors(real: [Double], imaginary: [Double], vectors: [Double]) -> MatrixDenseBLAS<Complex> {
        var result = MatrixDenseBLAS<Complex>(rows: rows, columns: columns, initialValue: .zero)
        var column = 0
        while column < columns {
            if imaginary[column] == 0.0 {
                for row in 0..<rows {
                    result[row, column] = Complex(vectors[row + rows * column], 0.0)
                }
                column += 1
            } else {
                for row in 0..<rows {
                    let realPart = vectors[row + rows * column]
                    let imaginaryPart = vectors[row + rows * (column + 1)]
                    result[row, column] = Complex(realPart, imaginaryPart)
                    result[row, column + 1] = Complex(realPart, -imaginaryPart)
                }
                column += 2
            }
        }
        return result
    }
}

infix operator * : MultiplicationPrecedence

public func * <M: PluMatrix, V: PluVector>(lhs: M, rhs: V) -> V where M.S == V.S { return lhs.times(rhs) }
public func * <L: PluMatrix, R: PluMatrix>(lhs: L, rhs: R) -> L where L.S == R.S { return lhs.times(rhs) }
