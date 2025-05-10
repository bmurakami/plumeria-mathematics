public protocol Vector {
    associatedtype Value
    var count: Int { get }
    subscript(i: Int) -> Value { get set }
}

public struct RealDenseVector: Vector {
    public typealias Value = Double
    
    public private(set) var values: [Double]
    
    public init(_ values: [Double]) {
        self.values = values
    }
    
    public var count: Int {
        return values.count
    }
    
    public subscript(i: Int) -> Double {
        get { return values[i] }
        set { values[i] = newValue }
    }
}
