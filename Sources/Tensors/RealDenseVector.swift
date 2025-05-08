struct RealDenseVector: VectorImplementation {
    typealias Scalar = Double
    
    private var values: [Double]
    let size: Int
    
    init(size: Int, initialValue: Double = 0.0) {
        self.size = size
        self.values = Array(repeating: 0.0, count: size)
    }
    
    init(_ values: [Double]) throws {
        func validate() throws {
            if values.isEmpty {
                throw MatrixError.malformedMatrix(reason: "The vector cannot be empty")
            }
        }
        
        try validate()
        self.size = values.count
        self.values = values
    }
    
    subscript(i: Int) -> Double {
        get { return values[i] }
        set { values[i] = newValue }
    }
}
