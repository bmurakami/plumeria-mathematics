public struct DenseMatrix<T>: Matrix {
    public private(set)var values: [[T]]

    public init(rows: Int, columns: Int, intialValue: T = 0.0) {
        values = Array(repeating: Array(repeating: intialValue, count: columns), count: rows)
    }

    public init(_ values: [[T]]) throws {
        func validate() throws {
            if values.isEmpty || (values.count > 0 && values[0].isEmpty) {
                throw MatrixError.malformedMatrix(reason: "The matrix cannot be empty or partially empty")
            }
            for row in values {
                if row.count != values[0].count {
                    throw MatrixError.malformedMatrix(reason: "All rows must have the same number of columns")
                }
            }
        }
        
        try validate()
        self.values = values
    }
    
    public func flatten() -> [T] {
        return Array(values.joined())
    }
    
    public var rows: Int { return values.count }
    public var columns: Int { return values[0].count }
    
    public var t: any Matrix {
        var At = DenseMatrix<T>(rows: self.rows, columns: self.columns, intialValue: values[0][0])
        for i in 0..<self.rows {
            for j in 0..<self.columns {
                At[j, i] = values[i][j]
            }
        }
        return At
    }
    
    public subscript(i: Int, j: Int) -> T {
        get { return values[i][j] }
        set { values[i][j] = newValue }
    }
}
