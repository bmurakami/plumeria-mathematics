public protocol PluVector: PluTensor, TensorStructure where S: PluScalar {
    var size: Int { get }
    subscript(index: Int) -> S { get set }
    
    init(_ elements: [S])
    
    func toArray(round: Bool) -> [S]
}

extension PluVector {
    public var shape: [Int] { [size] }
    public var rank: Int { 1 }

    public func toArray() -> [S] {
        return toArray(round: false)
    }
}

extension PluVector {
    public func magnitude() -> S.Magnitude {
        toArray().map { $0.magnitude * $0.magnitude }.reduce(.zero, +).squareRoot()
    }

    public func dot<V: PluVector>(_ other: V) -> S where V.S == S {
        precondition(size == other.size, "Vector sizes must match")
        var sum = S.zero
        for index in 0..<size {
            sum += self[index] * other[index]
        }
        return sum
    }

    public func cross<V: PluVector>(_ other: V) -> Self where V.S == S {
        precondition(size == 3 && other.size == 3, "Cross product requires 3D vectors")
        return Self([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0]
        ])
    }
}
