struct Matrix<T> {
    private var _implementation: any MatrixImplementation<T>
    
    var rows: Int { _implementation.rows }
    var columns: Int { _implementation.columns }
    
    init(_ values: [[T]]) throws where T == Double {
        self._implementation = try RealDenseMatrix(values)
    }
        
    init(implementation: any MatrixImplementation<T>) {
        self._implementation = implementation
    }
    
    var implementation: any MatrixImplementation<T> {
        get { _implementation }
        set { _implementation = newValue }
    }
    
    subscript(row: Int, column: Int) -> T {
        get {
            return _implementation[row, column]
        }
        set {
            var mutableImplementation = _implementation
            mutableImplementation[row, column] = newValue
            _implementation = mutableImplementation
        }
    }
}

protocol MatrixImplementation<Scalar> {
    associatedtype Scalar
    
    var rows: Int { get }
    var columns: Int { get }
    
    subscript(row: Int, column: Int) -> Scalar { get set }
}

enum MatrixError: Error {
    case malformedMatrix(reason: String)
}
