public protocol MatrixView: TensorView {
    var rows: Int { get }
    var columns: Int { get }

    subscript(row: Int, column: Int) -> Scalar { get set }
}
