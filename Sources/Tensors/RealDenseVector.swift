struct RealDenseVector: VectorImplementation {
    typealias Scalar = Double
    
    private var data: [Double]
    let size: Int
    
    init(size: Int, initialValue: Double = 0.0) {
        self.size = size
        self.data = Array(repeating: 0.0, count: size)
    }
    
    init(_ values: [Double]) throws {
        func validate() throws {
            if values.isEmpty {
                throw MatrixError.malformedMatrix(reason: "The vector cannot be empty")
            }
        }
        
        try validate()
        self.size = values.count
        self.data = values
    }
    
    subscript(i: Int) -> Double {
        get { return data[i] }
        set { data[i] = newValue }
    }
}
