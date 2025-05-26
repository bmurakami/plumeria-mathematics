//public struct DenseVector<S: Scalar>: Vector {
//    public private(set) var values: [S]
//    
//    // Need initializer for size.
//    
//    public init(_ values: [S]) {
//        self.values = values
//    }
//    
//    public static func + (lhs: DenseVector<S>, rhs: DenseVector<S>) -> DenseVector<S> {
//        guard lhs.size == rhs.size else {
//            fatalError("Vector dimensions must match for addition")
//        }
//        let result = zip(lhs.values, rhs.values).map { $0 + $1 }
//        return DenseVector(result)
//    }
//    
////    public static func - (lhs: DenseVector<S>, rhs: DenseVector<S>) -> DenseVector<S> {
////        guard lhs.size == rhs.size else {
////            fatalError("Vector dimensions must match for subtraction")
////        }
////        let result = zip(lhs.values, rhs.values).map { $0 - $1 }
////        return DenseVector(result)
////    }
//    
//    public var size: Int {
//        return values.count
//    }
//    
//    public subscript(i: Int) -> S {
//        get { return values[i] }
//        set { values[i] = newValue }
//    }
//    
//    public func toArray() -> [S] {
//        return values
//    }
//}
