public enum TensorError: Error {
    case incompatibleDimensions(got: Int, expected: Int)
    case indexOutOfBounds(dimension: Int, index: Int, bound: Int)
}

public protocol FlatTensor {
    associatedtype Scalar: PluScalar
    
    var shape: [Int] { get }
    
    subscript(_ indices: Int...) -> Scalar { get set }
    subscript(_ indices: [Int]) -> Scalar { get set }
}

public extension FlatTensor {
    var rank: Int { shape.count }
    var count: Int { shape.reduce(1, *) }
}

public struct Vector<Scalar: PluScalar>: FlatTensor {
    public let shape: [Int]
    var elements: [Scalar]
    let strides: [Int]
    
    public init(_ elements: [Scalar]) {
        let shape = [elements.count]
        self.shape = shape
        self.elements = elements
        self.strides = Self.computeStrides(shape: shape)
    }
    
    public init(count: Int, repeating value: Scalar) {
        let shape = [count]
        let data = Array(repeating: value, count: count)
        self.shape = shape
        self.elements = data
        self.strides = Self.computeStrides(shape: shape)
    }
    
    private static func computeStrides(shape: [Int]) -> [Int] {
        var strides = Array(repeating: 0, count: shape.count)
        if !shape.isEmpty {
            strides[shape.count - 1] = 1
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[i] = strides[i + 1] * shape[i + 1]
            }
        }
        return strides
    }
    
    private func linearIndex(indices: [Int]) throws -> Int {
        guard indices.count == shape.count else {
            throw TensorError.incompatibleDimensions(got: indices.count, expected: shape.count)
        }
        for (i, index) in indices.enumerated() {
            guard index >= 0 && index < shape[i] else {
                throw TensorError.indexOutOfBounds(dimension: i, index: index, bound: shape[i])
            }
        }
        return zip(indices, strides).map(*).reduce(0, +)
    }
    
    public subscript(_ indices: Int...) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    public subscript(_ indices: [Int]) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    public subscript(_ index: Int) -> Scalar {
        get { elements[try! linearIndex(indices: [index])] }
        set { elements[try! linearIndex(indices: [index])] = newValue }
    }
    
    public var description: String {
        return "\(elements)"
    }
}

public struct Matrix<Scalar: PluScalar>: FlatTensor {
    public let shape: [Int]
    var elements: [Scalar]
    let strides: [Int]
    
    public init(rows: Int, columns: Int, data: [Scalar]) {
        let shape = [rows, columns]
        precondition(shape.reduce(1, *) == data.count)
        self.shape = shape
        self.elements = data
        self.strides = Self.computeStrides(shape: shape)
    }
    
    public init(rows: Int, columns: Int, repeating value: Scalar) {
        let shape = [rows, columns]
        let data = Array(repeating: value, count: rows * columns)
        self.shape = shape
        self.elements = data
        self.strides = Self.computeStrides(shape: shape)
    }
    
    public var rows: Int { shape[0] }
    public var columns: Int { shape[1] }
    
    private static func computeStrides(shape: [Int]) -> [Int] {
        var strides = Array(repeating: 0, count: shape.count)
        if !shape.isEmpty {
            strides[shape.count - 1] = 1
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[i] = strides[i + 1] * shape[i + 1]
            }
        }
        return strides
    }
    
    private func linearIndex(indices: [Int]) throws -> Int {
        guard indices.count == shape.count else {
            throw TensorError.incompatibleDimensions(got: indices.count, expected: shape.count)
        }
        for (i, index) in indices.enumerated() {
            guard index >= 0 && index < shape[i] else {
                throw TensorError.indexOutOfBounds(dimension: i, index: index, bound: shape[i])
            }
        }
        return zip(indices, strides).map(*).reduce(0, +)
    }
    
    public subscript(_ indices: Int...) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    public subscript(_ indices: [Int]) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    public subscript(_ row: Int, _ column: Int) -> Scalar {
        get { elements[try! linearIndex(indices: [row, column])] }
        set { elements[try! linearIndex(indices: [row, column])] = newValue }
    }
    
    public var description: String {
        let rows = (0..<self.rows).map { row in
            (0..<columns).map { col in
                elements[row * strides[0] + col * strides[1]]
            }
        }
        return rows.map { "\($0)" }.joined(separator: "\n")
    }
}

public struct Tensor<Scalar: PluScalar>: FlatTensor {
    public let shape: [Int]
    var elements: [Scalar]
    let strides: [Int]
    
    public init(shape: [Int], data: [Scalar]) {
        precondition(shape.reduce(1, *) == data.count)
        self.shape = shape
        self.elements = data
        self.strides = Self.computeStrides(shape: shape)
    }
    
    public init(shape: [Int], repeating value: Scalar) {
        let data = Array(repeating: value, count: shape.reduce(1, *))
        self.shape = shape
        self.elements = data
        self.strides = Self.computeStrides(shape: shape)
    }
    
    private static func computeStrides(shape: [Int]) -> [Int] {
        var strides = Array(repeating: 0, count: shape.count)
        if !shape.isEmpty {
            strides[shape.count - 1] = 1
            for i in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[i] = strides[i + 1] * shape[i + 1]
            }
        }
        return strides
    }
    
    private func linearIndex(indices: [Int]) throws -> Int {
        guard indices.count == shape.count else {
            throw TensorError.incompatibleDimensions(got: indices.count, expected: shape.count)
        }
        for (i, index) in indices.enumerated() {
            guard index >= 0 && index < shape[i] else {
                throw TensorError.indexOutOfBounds(dimension: i, index: index, bound: shape[i])
            }
        }
        return zip(indices, strides).map(*).reduce(0, +)
    }
    
    public subscript(_ indices: Int...) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    public subscript(_ indices: [Int]) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    public var description: String {
        return "Tensor of shape: \(shape) and elements: \(elements))"
    }
}
