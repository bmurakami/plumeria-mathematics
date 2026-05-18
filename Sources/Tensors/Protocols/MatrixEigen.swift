public protocol MatrixEigen: PluMatrix where S == Double {
    associatedtype Eigenvectors: PluMatrix where Eigenvectors.S == Complex
    func eigen() -> Eigen<Eigenvectors>
}
