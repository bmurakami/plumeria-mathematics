public protocol TensorView {
    associatedtype Scalar: PluScalar
    
    var shape: [Int] { get }
    var rank: Int { get }
    var elements: [Scalar] { get }
    var isContiguous: Bool { get }
    
    subscript(_ indices: [Int]) -> Scalar { get set }
}
