public protocol VectorArithmeticReference: PluVector {}

extension VectorArithmeticReference {
    public func magnitude() -> S.Magnitude {
        toArray().map { $0.magnitude * $0.magnitude }.reduce(.zero, +).squareRoot()
    }
}
