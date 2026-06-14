public protocol MatrixView: TensorView {
    var rows: Int { get }
    var columns: Int { get }

    subscript(i: Int, j: Int) -> Scalar { get set }
}
