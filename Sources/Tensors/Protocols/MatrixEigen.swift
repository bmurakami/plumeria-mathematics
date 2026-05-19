public protocol MatrixEigen: PluMatrix where S == Double {
    associatedtype Eigenvalue: ComplexScalar
    associatedtype Eigenvectors: PluMatrix where Eigenvectors.S == Eigenvalue
    func eigen() -> Eigen<Eigenvalue, Eigenvectors>
}
