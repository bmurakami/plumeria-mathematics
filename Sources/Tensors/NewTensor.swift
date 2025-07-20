// ========================================
// MARK: - TensorError.swift
// ========================================

enum TensorError: Error {
    case incompatibleDimensions(got: Int, expected: Int)
    case indexOutOfBounds(dimension: Int, index: Int, bound: Int)
}

// ========================================
// MARK: - TensorProtocol.swift
// ========================================

protocol TensorProtocol {
    associatedtype Scalar: PluScalar
    var shape: [Int] { get }
    subscript(_ indices: Int...) -> Scalar { get set }
    subscript(_ indices: [Int]) -> Scalar { get set }
}

extension TensorProtocol {
    var rank: Int { shape.count }
    var elementCount: Int { shape.reduce(1, *) }
}

protocol BLASCompatible {
    associatedtype Scalar: PluScalar
    var elements: [Scalar] { get }
}

extension BLASCompatible {
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) -> R) -> R {
        return elements.withUnsafeBufferPointer(body)
    }
}

// ========================================
// MARK: - Tensor.swift
// ========================================

struct Tensor<Scalar: PluScalar> {
    private var elements: [Scalar]
    let shape: [Int]
    private let strides: [Int]
    
    init(shape: [Int], data: [Scalar]) {
        precondition(shape.reduce(1, *) == data.count)
        self.shape = shape
        self.elements = data
        self.strides = Self.computeStrides(shape: shape)
    }
    
    init(shape: [Int], repeating value: Scalar) {
        let count = shape.reduce(1, *)
        self.shape = shape
        self.elements = Array(repeating: value, count: count)
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
    
    subscript(_ indices: Int...) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    subscript(_ indices: [Int]) -> Scalar {
        get { elements[try! linearIndex(indices: indices)] }
        set { elements[try! linearIndex(indices: indices)] = newValue }
    }
    
    subscript(_ index: Int) -> TensorSlice<Scalar> {
        precondition(shape.count > 0, "Cannot slice scalar tensor")
        let newShape = Array(shape.dropFirst())
        let newStrides = Array(strides.dropFirst())
        let sliceOffset = index * strides[0]
        
        return TensorSlice(
            elements: elements,
            offset: sliceOffset,
            shape: newShape,
            strides: newStrides
        )
    }
}

extension Tensor: TensorProtocol {}

extension Tensor {
    var description: String {
        return "shape: \(shape), elements: \(elements)"
    }
}

// ========================================
// MARK: - TensorSlice.swift
// ========================================

struct TensorSlice<Scalar: PluScalar> {
    private let elements: [Scalar]
    private let offset: Int
    let shape: [Int]
    private let strides: [Int]
    
    init(elements: [Scalar], offset: Int, shape: [Int], strides: [Int]) {
        self.elements = elements
        self.offset = offset
        self.shape = shape
        self.strides = strides
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
    
    subscript(_ indices: Int...) -> Scalar {
        elements[offset + (try! linearIndex(indices: indices))]
    }
    
    subscript(_ indices: [Int]) -> Scalar {
        elements[offset + (try! linearIndex(indices: indices))]
    }
    
    subscript(_ index: Int) -> TensorSlice<Scalar> {
        precondition(shape.count > 0, "Cannot slice scalar tensor")
        let newShape = Array(shape.dropFirst())
        let newStrides = Array(strides.dropFirst())
        let sliceOffset = offset + index * strides[0]
        
        return TensorSlice(
            elements: elements,
            offset: sliceOffset,
            shape: newShape,
            strides: newStrides
        )
    }
}

extension TensorSlice {
    var description: String {
        let elementValues = (0..<shape.reduce(1, *)).map { i in
            let coords = indexToCoordinates(i, shape: shape)
            return elements[offset + zip(coords, strides).map(*).reduce(0, +)]
        }
        return "shape: \(shape), elements: \(elementValues)"
    }
    
    private func indexToCoordinates(_ index: Int, shape: [Int]) -> [Int] {
        var coords = Array(repeating: 0, count: shape.count)
        var remaining = index
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            let dimSize = shape[i]
            coords[i] = remaining % dimSize
            remaining /= dimSize
        }
        return coords
    }
}
