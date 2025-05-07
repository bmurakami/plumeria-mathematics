struct RealDenseMatrix: MatrixImplementation {
    typealias Scalar = Double
    
    private var data: [[Double]]
    let rows: Int
    let columns: Int
    
    init(rows: Int, columns: Int, initialValue: Double = 0.0) {
        self.rows = rows
        self.columns = columns
        self.data = Array(repeating: Array(repeating: initialValue, count: columns), count: rows)
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
        self.rows = values.count
        self.columns = values[0].count
        self.data = values
    }
    
    subscript(row: Int, column: Int) -> Double {
        get { return data[row][column] }
        set { data[row][column] = newValue }
    }
}
