public protocol TensorView: TensorStructure where S == Scalar {
    associatedtype Scalar: PluScalar
    
    var elements: [Scalar] { get }
    var isContiguous: Bool { get }
    
    subscript(_ indices: [Int]) -> Scalar { get set }
}
