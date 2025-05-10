public protocol Vector {
    associatedtype Value
    
    var count: Int { get }
    subscript(i: Int) -> Value { get set }
}

public struct DenseVector<T>: Vector {
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
