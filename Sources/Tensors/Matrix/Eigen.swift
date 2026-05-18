public struct Eigen<Vectors: PluMatrix>: Equatable where Vectors.S == Complex {
    public let values: [Complex]
    public let vectors: Vectors

    public init(values: [Complex], vectors: Vectors) {
        self.values = values
        self.vectors = vectors
    }
}
