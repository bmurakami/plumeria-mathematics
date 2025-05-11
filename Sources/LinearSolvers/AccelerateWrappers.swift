#if os(macOS)
import Accelerate
import Tensors

func solveDenseRealLinear_Accelerate(_ A: DenseMatrix<Double>, _ b: DenseVector<Double>) -> DenseVector<Double> {
    precondition(A.rows == A.columns, "Matrix A must be square")
    precondition(A.rows == b.count, "Matrix A and vector b must have compatible dimensions")
    
    var n = Int32(A.rows)
    var nrhs = Int32(1)
    var matrixA = (A.t as! DenseMatrix<Double>).flatten()
    var bCopy = Array(b.values)
    var lda = Int32(A.rows)
    var ldb = Int32(A.rows)
    var ipiv = [Int32](repeating: 0, count: A.rows)
    var info = Int32(0)
    
    Accelerate.dgesv_(&n, &nrhs, &matrixA, &lda, &ipiv, &bCopy, &ldb, &info)
    
    return DenseVector(bCopy)
}
#endif
