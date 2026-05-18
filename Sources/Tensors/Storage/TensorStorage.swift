public final class TensorStorage<Scalar: PluScalar> {
    public var elements: [Scalar]

    public init(_ elements: [Scalar]) {
        self.elements = elements
    }

    public subscript(index: Int) -> Scalar {
        get { elements[index] }
        set { elements[index] = newValue }
    }
}
