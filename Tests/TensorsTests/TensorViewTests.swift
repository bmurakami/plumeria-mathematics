import Testing
@testable import Tensors

@Test func TensorView_freshVectorHasColumnMajorStride() {
    let view = TensorView<Double>(shape: [4])
    
    #expect(view.offset == 0)
    #expect(view.shape == [4])
    #expect(view.strides == [1])
    #expect(view.rank == 1)
    #expect(view.count == 4)
    #expect(view.isContiguous)
    #expect(view.storage.elements == [0.0, 0.0, 0.0, 0.0])
}

@Test func TensorView_freshMatrixHasColumnMajorStrides() {
    let view = TensorView<Double>(shape: [2, 3])
    
    #expect(view.offset == 0)
    #expect(view.shape == [2, 3])
    #expect(view.strides == [1, 2])
    #expect(view.rank == 2)
    #expect(view.count == 6)
    #expect(view.isContiguous)
    #expect(view.storage.elements == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

@Test func TensorView_readsColumnMajorElements() {
    let view = TensorView<Double>(
        shape: [2, 3],
        elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    )
    
    #expect(view[[0, 0]] == 1.0)
    #expect(view[[1, 0]] == 4.0)
    #expect(view[[0, 1]] == 2.0)
    #expect(view[[1, 1]] == 5.0)
    #expect(view[[0, 2]] == 3.0)
    #expect(view[[1, 2]] == 6.0)
}

@Test func TensorView_subscriptMutationUpdatesElement() {
    var view = TensorView<Double>(shape: [2, 2], elements: [1.0, 3.0, 2.0, 4.0])
    view[[1, 0]] = 99.0
    
    #expect(view[[1, 0]] == 99.0)
    #expect(view.storage.elements == [1.0, 99.0, 2.0, 4.0])
}

@Test func TensorView_copiedViewsShareStorage() {
    let original = TensorView<Double>(shape: [3], elements: [1.0, 2.0, 3.0])
    let copy = original
    
    #expect(original.storage === copy.storage)
}

@Test func TensorView_mutatingCopiedViewDetachesStorage() {
    let original = TensorView<Double>(shape: [3], elements: [1.0, 2.0, 3.0])
    var copy = original
    
    copy[[1]] = 99.0
    
    #expect(original.storage !== copy.storage)
    #expect(original[[1]] == 2.0)
    #expect(copy[[1]] == 99.0)
}

@Test func TensorView_mutatingUniqueViewDoesNotCopy() {
    var view = TensorView<Double>(shape: [3], elements: [1.0, 2.0, 3.0])
    let storageID = ObjectIdentifier(view.storage)
    
    view[[1]] = 99.0
    
    #expect(ObjectIdentifier(view.storage) == storageID)
    #expect(view.storage.elements == [1.0, 99.0, 3.0])
}

@Test func TensorView_canRepresentOffsetAndStridedView() {
    let storage = TensorStorage([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let view = TensorView(storage: storage, offset: 1, shape: [3], strides: [2])
    
    #expect(view[[0]] == 2.0)
    #expect(view[[1]] == 4.0)
    #expect(view[[2]] == 6.0)
}

@Test("Contiguous and non-contiguous views", arguments: [
    ([1.0, 2.0, 3.0], 0, [3], [1], true),
    ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, [2], [1], true),
    ([1.0, 2.0, 3.0, 4.0, 5.0], 0, [3], [2], false),
    ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1, [3], [2], false)
])
func TensorView_contiguity(elements: [Double], offset: Int, shape: [Int], strides: [Int], expected: Bool) {
    let storage = TensorStorage(elements)
    let view = TensorView(storage: storage, offset: offset, shape: shape, strides: strides)
    
    #expect(view.isContiguous == expected)
}

@Test func TensorView_scalarViewIsContiguous() {
    let view = TensorView<Double>(shape: [], elements: [42.0])
    
    #expect(view.isContiguous)
}

@Test func TensorView_emptyDimensionViewIsContiguousWithColumnMajorStrides() {
    let view = TensorView<Double>(shape: [0, 3])
    
    #expect(view.strides == [1, 0])
    #expect(view.isContiguous)
}

@Test func TensorView_readOnlyAccessDoesNotCopyStorage() {
    let storage = TensorStorage([1.0, 2.0, 3.0])
    let view = TensorView(storage: storage, offset: 0, shape: [3], strides: [1])
    
    _ = view[[1]]
    
    #expect(view.storage === storage)
}
