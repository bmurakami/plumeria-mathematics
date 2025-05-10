public protocol Vector {
    associatedtype Value
    
    var count: Int { get }
    subscript(i: Int) -> Value { get set }
}

