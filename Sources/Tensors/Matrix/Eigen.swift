public struct Eigen<Value: ComplexScalar, Vectors: PluMatrix>: Equatable where Vectors.S == Value {
    public let values: [Value]
    public let vectors: Vectors

    public init(values: [Value], vectors: Vectors) {
        self.values = values
        self.vectors = vectors
    }
}
