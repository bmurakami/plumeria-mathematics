public protocol VectorView: TensorView {
    var size: Int { get }
    
    subscript(index: Int) -> Scalar { get set }
}
