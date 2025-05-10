struct DenseMatrix<T>: Matrix {
    public var rows: Int { return values.count }
    public var columns: Int { return values[0].count }
    private var values: [[T]]

    init(rows: Int, columns: Int, intialValue: T = 0.0) {
        values = Array(repeating: Array(repeating: intialValue, count: columns), count: rows)
    }
    
    init(_ values: [[T]]) throws {
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
    
    subscript(i: Int, j: Int) -> T {
        get { return values[i][j] }
        set { values[i][j] = newValue }
    }
}
