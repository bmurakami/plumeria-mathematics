public protocol PluVector: PluTensor, TensorStructure where S: PluScalar {
    var size: Int { get }
    subscript(i: Int) -> S { get set }

    init(_ elements: [S])

    func toArray(round: Bool) -> [S]
    func magnitude() -> S.Magnitude
}

extension PluVector {
    public var shape: [Int] { [size] }
    public var rank: Int { 1 }

    public func toArray() -> [S] {
        return toArray(round: false)
    }

    public subscript(range: Range<Int>) -> Self {
        get { self[TensorSliceIndex.range(range)] }
        set { self[TensorSliceIndex.range(range)] = newValue }
    }

    public subscript(i: TensorSliceIndex) -> Self {
        get {
            let range = i.sliceRange(dimensionSize: size)
            var result = Self(Array(repeating: .zero, count: range.length))
            for position in 0..<range.length {
                result[position] = self[range.start + position * range.step]
            }
            return result
        }
        set {
            let range = i.sliceRange(dimensionSize: size)
            let error = sliceAssignmentShapeError(destination: [range.length], replacement: newValue.shape)
            if let error {
                preconditionFailure(error)
            }
            for position in 0..<range.length {
                self[range.start + position * range.step] = newValue[position]
            }
        }
    }
}

extension PluVector {
    public func dot<V: PluVector>(_ other: V) -> S where V.S == S {
        precondition(size == other.size, "Vector sizes must match")
        var sum = S.zero
        for i in 0..<size {
            sum += self[i] * other[i]
        }
        return sum
    }

    public func cross<V: PluVector>(_ other: V) -> Self where V.S == S {
        precondition(size == 3 && other.size == 3, "Cross product requires 3D vectors")
        let x = self[1] * other[2] - self[2] * other[1]
        let y = self[2] * other[0] - self[0] * other[2]
        let z = self[0] * other[1] - self[1] * other[0]
        return Self([x, y, z])
    }
}
