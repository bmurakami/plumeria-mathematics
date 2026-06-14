public final class TensorStorage<Scalar: PluScalar> {
    public var elements: [Scalar]

    public init(_ elements: [Scalar]) {
        self.elements = elements
    }

    public subscript(i: Int) -> Scalar {
        get { elements[i] }
        set { elements[i] = newValue }
    }
}
