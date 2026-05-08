public enum TensorStorageOrder {
    case columnMajor
    case rowMajor
}

public protocol FlatTensor {
    associatedtype Scalar: PluScalar
    
    var elements: [Scalar] { get set }
    var rank: Int { get }
    var shape: [Int] { get }

    init(shape: [Int])
    init(shape: [Int], elements: [Scalar])
    
    subscript(indices indices: Int...) -> Scalar { get set }
    subscript(_ indices: [Int]) -> Scalar { get set }
    
    func flatten(order: TensorStorageOrder) -> [Scalar]
}

public extension FlatTensor {
    var rank: Int { shape.count }
    var count: Int { shape.reduce(1, *) }
    
    private var strides: [Int] {
        var strides = Array(repeating: 0, count: shape.count)
        if !shape.isEmpty {
            strides[0] = 1
            for i in 1..<shape.count {
                strides[i] = strides[i - 1] * shape[i - 1]
            }
        }
        return strides
    }
    
    private func linearIndex(indices: [Int]) -> Int {
        precondition(
            indices.count == shape.count,
            "Tensor index rank \(indices.count) does not match tensor rank \(shape.count)"
        )
        for (i, index) in indices.enumerated() {
            precondition(
                index >= 0 && index < shape[i],
                "Tensor index \(index) is out of bounds for dimension \(i) with size \(shape[i])"
            )
        }
        
        return zip(indices, strides).map(*).reduce(0, +)
    }
    
    subscript(indices indices: Int...) -> Scalar {
        get { elements[linearIndex(indices: indices)] }
        set { elements[linearIndex(indices: indices)] = newValue }
    }
    
    subscript(_ indices: [Int]) -> Scalar {
        get { elements[linearIndex(indices: indices)] }
        set { elements[linearIndex(indices: indices)] = newValue }
    }
    
    func flatten(order: TensorStorageOrder = .columnMajor) -> [Scalar] {
        switch order {
        case .columnMajor:
            return elements
        case .rowMajor:
            guard !shape.isEmpty else { return elements }
            
            return (0..<count).map { rowMajorIndex in
                var remainder = rowMajorIndex
                var indices = Array(repeating: 0, count: shape.count)
                
                for dimension in stride(from: shape.count - 1, through: 0, by: -1) {
                    indices[dimension] = remainder % shape[dimension]
                    remainder /= shape[dimension]
                }
                return self[indices]
            }
        }
    }
    
    fileprivate static func validatedElementCount(for shape: [Int], elements: [Scalar]) -> Int {
        precondition(shape.allSatisfy { $0 >= 0 }, "Tensor shape dimensions must be non-negative")
        
        let count = shape.reduce(1, *)
        precondition(count == elements.count, "Tensor shape \(shape) requires \(count) elements, but got \(elements.count)")
        return count
    }
}

public struct FlatMatrix<Scalar: PluScalar>: FlatTensor {
    public let shape: [Int]
    public var elements: [Scalar]
    
    public init(shape: [Int]) {
        precondition(shape.count == 2, "Matrix shape must have rank 2")
        precondition(shape.allSatisfy { $0 >= 0 }, "Matrix shape dimensions must be non-negative")
        
        self.shape = shape
        self.elements = Array(repeating: .zero, count: shape.reduce(1, *))
    }
    
    public init(shape: [Int], elements: [Scalar]) {
        precondition(shape.count == 2, "Matrix shape must have rank 2")
        _ = Self.validatedElementCount(for: shape, elements: elements)
        
        self.shape = shape
        self.elements = elements
    }
    
    public init(rows: Int, columns: Int, data: [Scalar]) {
        self.init(shape: [rows, columns], elements: data)
    }
    
    public init(rows: Int, columns: Int, repeating value: Scalar) {
        precondition(rows >= 0 && columns >= 0, "Matrix dimensions must be non-negative")
        
        self.shape = [rows, columns]
        self.elements = Array(repeating: value, count: rows * columns)
    }
    
    public var rows: Int { shape[0] }
    public var columns: Int { shape[1] }
    
    public subscript(_ row: Int, _ column: Int) -> Scalar {
        get { self[[row, column]] }
        set { self[[row, column]] = newValue }
    }
    
    public var description: String {
        let rows = (0..<self.rows).map { row in
            (0..<columns).map { col in
                self[row, col]
            }
        }
        return rows.map { "\($0)" }.joined(separator: "\n")
    }
}
