public protocol MatrixEigen: PluMatrix where S == Double {
    func eigen() -> Eigen
}
