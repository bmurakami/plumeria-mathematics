#if canImport(Accelerate)
import AccelerateWrapper
#endif
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
        #if canImport(Accelerate)
        case .accelerate:
            let _ = AccelerateOperations.dgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        #endif
        case .openBLAS:
            let _ = OpenBLASOperations.dgesv(Int32(n), &AArray, &bArray)
            x = bArray as! [V.S]
        }
    case is Complex.Type:
        var AArray = interleaved(A.flatten() as! [Complex])
        var bArray = interleaved(b.toArray() as! [Complex])

        switch blasImplementation {
        #if canImport(Accelerate)
        case .accelerate:
            let _ = AccelerateOperations.zgesv(Int32(n), &AArray, &bArray)
            x = complexValues(bArray) as! [V.S]
        #endif
        case .openBLAS:
            let _ = OpenBLASOperations.zgesv(Int32(n), &AArray, &bArray)
            x = complexValues(bArray) as! [V.S]
        }

    default:
        fatalError("Unsupported scalar type")
    }

    return V(x)
}

private func interleaved(_ values: [Complex]) -> [Double] {
    values.flatMap { [$0.real, $0.imaginary] }
}

private func complexValues(_ values: [Double]) -> [Complex] {
    stride(from: 0, to: values.count, by: 2).map { Complex(values[$0], values[$0 + 1]) }
}
