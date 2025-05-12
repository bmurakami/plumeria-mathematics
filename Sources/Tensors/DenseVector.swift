public struct DenseVector<T: FloatingPoint & ApproximatelyEquatable>: Vector {
    public private(set) var values: [T]
    
    public init(_ values: [T]) {
        self.values = values
    }
    
    public var count: Int {
        return values.count
    }
    
    public subscript(i: Int) -> T {
        get { return values[i] }
        set { values[i] = newValue }
    }
}
