import Testing
@testable import Tensors

@Test func TensorDenseReference_infersShapeFromNestedArrays() {
    let tensor = TensorDenseReference<Double>([
        [[1.0, -1.0], [2.0, 0.0]],
        [[0.0, 2.0], [-2.0, 1.0]],
        [[3.0, 1.0], [1.0, -3.0]]
    ])

    #expect(tensor.shape == [3, 2, 2])
    #expect(tensor.rank == 3)
    #expect(tensor[0, 0, 0] == 1.0)
    #expect(tensor[1, 1, 0] == -2.0)
    #expect(tensor[2, 1, 1] == -3.0)
}
