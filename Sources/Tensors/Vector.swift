struct Vector<T> {
    private var _implementation: any VectorImplementation<T>
    
    var size: Int { _implementation.size }
    
    init(_ values: [T]) throws where T == Double {
        self._implementation = try RealDenseVector(values)
    }
        
    init(implementation: any VectorImplementation<T>) {
        self._implementation = implementation
    }
    
    var implementation: any VectorImplementation<T> {
        get { _implementation }
        set { _implementation = newValue }
    }
    
    subscript(i: Int) -> T {
        get {
            return _implementation[i]
        }
        set {
            var mutableImplementation = _implementation
            mutableImplementation[i] = newValue
            _implementation = mutableImplementation
        }
    }
}

protocol VectorImplementation<Scalar> {
    associatedtype Scalar
    
    var size: Int { get }

    subscript(i: Int) -> Scalar { get set }
}
