public protocol VectorView: TensorView {
    var size: Int { get }

    subscript(i: Int) -> Scalar { get set }
}
