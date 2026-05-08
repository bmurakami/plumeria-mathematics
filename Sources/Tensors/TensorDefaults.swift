public typealias Vector<S: PluScalar> = VectorBase<VectorDenseReference<S>>
public typealias Matrix<S: PluScalar> = MatrixBase<MatrixDenseBLAS<S>>

public typealias DenseVectorD = Vector<Double>
public typealias DenseMatrixD = Matrix<Double>
public typealias ReferenceVectorD = VectorBase<VectorDenseReference<Double>>
public typealias ReferenceMatrixD = MatrixBase<MatrixDenseReference<Double>>
