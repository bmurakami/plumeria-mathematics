public typealias Vector<S: PluScalar> = VectorBase<VectorDenseReference<S>>
public typealias Matrix<S: PluScalar> = MatrixBase<MatrixDenseBLAS<S>>

public typealias DenseVectorDouble = Vector<Double>
public typealias DenseMatrixDouble = Matrix<Double>
public typealias ReferenceVectorDouble = VectorBase<VectorDenseReference<Double>>
public typealias ReferenceMatrixDouble = MatrixBase<MatrixDenseReference<Double>>
