public struct Eigen: Equatable {
    public let values: [Complex]
    public let vectors: MatrixDenseBLAS<Complex>

    public init(values: [Complex], vectors: MatrixDenseBLAS<Complex>) {
        self.values = values
        self.vectors = vectors
    }
}
