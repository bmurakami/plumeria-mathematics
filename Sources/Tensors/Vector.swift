public protocol Vector : Equatable {
    associatedtype Scalar : Equatable
    
    var count: Int { get }
    subscript(i: Int) -> Scalar { get set }
}

extension Vector where Scalar: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        guard lhs.count == rhs.count else { return false }
        
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] {
                return false
            }
        }
        return true
    }
}

extension DenseVector where T: Equatable {
    public static func == (lhs: DenseVector<T>, rhs: [T]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] {
                return false
            }
        }
        return true
    }
    
    public static func == (lhs: [T], rhs: DenseVector<T>) -> Bool {
        return rhs == lhs
    }
}
