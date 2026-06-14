public protocol PluMatrix: PluTensor, TensorStructure, MatrixArithmetic where S: PluScalar {
    var rows: Int { get }
    var columns: Int { get }
    subscript(i: Int, j: Int) -> S { get set }

    init(rows: Int, columns: Int, initialValue: S)
    init(_ elements: [[S]])

    func toArray(round: Bool) -> [[S]]
    func flatten(columnMajorOrder: Bool) -> [S]
}

extension PluMatrix {
    public var shape: [Int] { [rows, columns] }
    public var rank: Int { 2 }
    public var t: Self { transpose() }
}

extension PluMatrix {
    public func toArray() -> [[S]] { return toArray(round: false) }
    public func flatten() -> [S] { return flatten(columnMajorOrder: true) }
    public static func identity(size: Int) -> Self {
        precondition(size > 0, "Identity matrix size must be positive")
        return Self((0..<size).map { row in
            (0..<size).map { column in row == column ? 1 : 0 }
        })
    }

    public subscript(rows: Range<Int>, columns: Range<Int>) -> Self {
        get { self[TensorSliceIndex.range(rows), TensorSliceIndex.range(columns)] }
        set { self[TensorSliceIndex.range(rows), TensorSliceIndex.range(columns)] = newValue }
    }

    public subscript(rows: Range<Int>, columns: TensorSliceIndex) -> Self {
        get { self[TensorSliceIndex.range(rows), columns] }
        set { self[TensorSliceIndex.range(rows), columns] = newValue }
    }

    public subscript(rows: TensorSliceIndex, columns: Range<Int>) -> Self {
        get { self[rows, TensorSliceIndex.range(columns)] }
        set { self[rows, TensorSliceIndex.range(columns)] = newValue }
    }

    public subscript(rows: TensorSliceIndex, columns: TensorSliceIndex) -> Self {
        get {
            let rowRange = rows.sliceRange(dimensionSize: self.rows)
            let columnRange = columns.sliceRange(dimensionSize: self.columns)
            var result = Self(rows: rowRange.length, columns: columnRange.length, initialValue: .zero)
            for i in 0..<rowRange.length {
                for j in 0..<columnRange.length {
                    let sourceI = rowRange.start + i * rowRange.step
                    let sourceJ = columnRange.start + j * columnRange.step
                    result[i, j] = self[sourceI, sourceJ]
                }
            }
            return result
        }
        set {
            let rowRange = rows.sliceRange(dimensionSize: self.rows)
            let columnRange = columns.sliceRange(dimensionSize: self.columns)
            let destinationShape = [rowRange.length, columnRange.length]
            let error = sliceAssignmentShapeError(destination: destinationShape, replacement: newValue.shape)
            if let error {
                preconditionFailure(error)
            }
            for i in 0..<rowRange.length {
                for j in 0..<columnRange.length {
                    let destinationI = rowRange.start + i * rowRange.step
                    let destinationJ = columnRange.start + j * columnRange.step
                    self[destinationI, destinationJ] = newValue[i, j]
                }
            }
        }
    }
}
