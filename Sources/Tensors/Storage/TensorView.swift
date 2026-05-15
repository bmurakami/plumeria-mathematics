public protocol TensorView: TensorStructure {
    associatedtype Scalar: PluScalar
    
    var elements: [Scalar] { get }
    var isContiguous: Bool { get }
    
    subscript(_ indices: [Int]) -> Scalar { get set }
}
