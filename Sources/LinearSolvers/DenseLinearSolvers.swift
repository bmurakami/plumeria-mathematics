import AccelerateWrapper
import OpenBLASWrapper
import Tensors

public func solveLinearDense<M: PluMatrix, V: PluVector>(_ A: M, _ b: V, blasImplementation: BLAS = .default) -> V {
    precondition(A.columns == b.size, "Number of columns in A must equal size of b")
    
    let n = b.size
    var x: [V.S]

    switch V.S.self {
    case is Double.Type:
        var AArray = A.flatten() as! [Double]
        var bArray = b.toArray() as! [Double]

        switch blasImplementation {
        case .accelerate:
            let _ = AccelerateOperations.dgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        case .openBLAS:
            let _ = OpenBLASOperations.dgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        }
    case is Complex.Type:
        fatalError("Not yet implemented")

    default:
        fatalError("Unsupported scalar type")
    }

    return V(x)
}
