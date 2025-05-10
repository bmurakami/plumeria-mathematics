struct RealDenseMatrix: Matrix {
    typealias Scalar = Double
    
    public var rows: Int { return values.count }
    public var columns: Int { return values[0].count }
    private var values: [[Double]]

    init(rows: Int, columns: Int) {
        values = Array(repeating: Array(repeating: 0.0, count: columns), count: rows)
    }
    
    init(_ values: [[Double]]) throws {
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
    
    subscript(i: Int, j: Int) -> Double {
        get { return values[i][j] }
        set { values[i][j] = newValue }
    }
}
