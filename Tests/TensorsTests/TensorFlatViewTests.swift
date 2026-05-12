import Testing
@testable import Tensors

@Test func TensorFlatView_freshVectorHasColumnMajorStride() {
    let view = TensorFlatView<Double>(shape: [4])
    
    #expect(view.offset == 0)
    #expect(view.shape == [4])
    #expect(view.strides == [1])
    #expect(view.rank == 1)
    #expect(view.count == 4)
    #expect(view.isContiguous)
    #expect(view.storage.elements == [0.0, 0.0, 0.0, 0.0])
}

@Test func TensorFlatView_freshMatrixHasColumnMajorStrides() {
    let view = TensorFlatView<Double>(shape: [2, 3])
    
    #expect(view.offset == 0)
    #expect(view.shape == [2, 3])
    #expect(view.strides == [1, 2])
    #expect(view.rank == 2)
    #expect(view.count == 6)
    #expect(view.isContiguous)
    #expect(view.storage.elements == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

@Test func TensorFlatView_readsColumnMajorElements() {
    let view = TensorFlatView<Double>(shape: [2, 3], elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    
    #expect(view[[0, 0]] == 1.0)
    #expect(view[[1, 0]] == 4.0)
    #expect(view[[0, 1]] == 2.0)
    #expect(view[[1, 1]] == 5.0)
    #expect(view[[0, 2]] == 3.0)
    #expect(view[[1, 2]] == 6.0)
}

@Test func TensorFlatView_subscriptMutationUpdatesElement() {
    var view = TensorFlatView<Double>(shape: [2, 2], elements: [1.0, 3.0, 2.0, 4.0])
    view[[1, 0]] = 99.0
    
    #expect(view[[1, 0]] == 99.0)
    #expect(view.storage.elements == [1.0, 99.0, 2.0, 4.0])
}

@Test func TensorFlatView_copiedViewsShareStorage() {
    let original = TensorFlatView<Double>(shape: [3], elements: [1.0, 2.0, 3.0])
    let copy = original
    
    #expect(original.storage === copy.storage)
}

@Test func TensorFlatView_mutatingCopiedViewDetachesStorage() {
    let original = TensorFlatView<Double>(shape: [3], elements: [1.0, 2.0, 3.0])
    var copy = original
    
    copy[[1]] = 99.0
    
    #expect(original.storage !== copy.storage)
    #expect(original[[1]] == 2.0)
    #expect(copy[[1]] == 99.0)
}

@Test func TensorFlatView_mutatingUniqueViewDoesNotCopy() {
    var view = TensorFlatView<Double>(shape: [3], elements: [1.0, 2.0, 3.0])
    let storageID = ObjectIdentifier(view.storage)
    
    view[[1]] = 99.0
    
    #expect(ObjectIdentifier(view.storage) == storageID)
    #expect(view.storage.elements == [1.0, 99.0, 3.0])
}

@Test func TensorFlatView_canRepresentOffsetAndStridedView() {
    let storage = TensorStorage([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let view = TensorFlatView(storage: storage, offset: 1, shape: [3], strides: [2])
    
    #expect(view[[0]] == 2.0)
    #expect(view[[1]] == 4.0)
    #expect(view[[2]] == 6.0)
}

@Test("Vector slicing", arguments: [
    (SliceRange(1..<4), [3], [1], true, [2.0, 3.0, 4.0]),
    (SliceRange(1..<6, step: 2), [3], [2], false, [2.0, 4.0, 6.0])
])
func TensorFlatView_slicesVector(range: SliceRange, shape: [Int], strides: [Int],
                                 isContiguous: Bool, values: [Double]) {
    let view = TensorFlatView<Double>(shape: [6], elements: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let slice = view.slice(range)
    
    #expect(slice.storage === view.storage)
    #expect(slice.offset == 1)
    #expect(slice.shape == shape)
    #expect(slice.strides == strides)
    #expect(slice.isContiguous == isContiguous)
    #expect((0..<slice.count).map { slice[[$0]] } == values)
}

@Test func TensorFlatView_slicesSubmatrix() {
    // Original:                   Slice rows 1..<3, columns 1..<3:
    // [1.0,  2.0,  3.0,  4.0]     [ 6.0,  7.0]
    // [5.0,  6.0,  7.0,  8.0]     [10.0, 11.0]
    // [9.0, 10.0, 11.0, 12.0]
    let view = TensorFlatView<Double>(
        shape: [3, 4],
        elements: [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0]
    )
    let slice = view.slice(rows: SliceRange(1..<3), columns: SliceRange(1..<3))
    
    #expect(slice.storage === view.storage)
    #expect(slice.offset == 4)
    #expect(slice.shape == [2, 2])
    #expect(slice.strides == [1, 3])
    #expect(!slice.isContiguous)
    #expect(slice[[0, 0]] == 6.0)
    #expect(slice[[1, 0]] == 10.0)
    #expect(slice[[0, 1]] == 7.0)
    #expect(slice[[1, 1]] == 11.0)
}

@Test func TensorFlatView_mutatingSliceCopiesOnWrite() {
    let view = TensorFlatView<Double>(shape: [4], elements: [1.0, 2.0, 3.0, 4.0])
    var slice = view.slice([SliceRange(1..<3)])
    
    slice[[0]] = 99.0
    
    #expect(slice.storage !== view.storage)
    #expect(view[[1]] == 2.0)
    #expect(slice[[0]] == 99.0)
}

@Test func TensorFlatView_chainsVectorSlices() {
    let view = TensorFlatView<Double>(shape: [8], elements: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    let firstSlice = view.slice(SliceRange(1..<8, step: 2))
    let secondSlice = firstSlice.slice(SliceRange(1..<3))
    
    #expect(secondSlice.storage === view.storage)
    #expect(secondSlice.offset == 3)
    #expect(secondSlice.shape == [2])
    #expect(secondSlice.strides == [2])
    #expect(!secondSlice.isContiguous)
    #expect(secondSlice[[0]] == 4.0)
    #expect(secondSlice[[1]] == 6.0)
}

@Test func TensorFlatView_chainsMatrixSlices() {
    // Original:                   First slice:       Second slice:
    // [1.0,  2.0,  3.0,  4.0]     [ 2.0,  3.0]       [ 3.0]
    // [5.0,  6.0,  7.0,  8.0]     [ 6.0,  7.0]       [ 7.0]
    // [9.0, 10.0, 11.0, 12.0]     [10.0, 11.0]
    let view = TensorFlatView<Double>(
        shape: [3, 4],
        elements: [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0]
    )
    let firstSlice = view.slice(rows: SliceRange(0..<3), columns: SliceRange(1..<3))
    let secondSlice = firstSlice.slice(rows: SliceRange(0..<2), columns: SliceRange(1..<2))
    
    #expect(secondSlice.storage === view.storage)
    #expect(secondSlice.offset == 6)
    #expect(secondSlice.shape == [2, 1])
    #expect(secondSlice.strides == [1, 3])
    #expect(!secondSlice.isContiguous)
    #expect(secondSlice[[0, 0]] == 3.0)
    #expect(secondSlice[[1, 0]] == 7.0)
}

@Test func TensorFlatView_convenienceSlicesMatchCoreSlice() {
    let vector = TensorFlatView<Double>(shape: [4], elements: [1.0, 2.0, 3.0, 4.0])
    let matrix = TensorFlatView<Double>(shape: [2, 3], elements: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    let vectorSlice = vector.slice(SliceRange(1..<3))
    let coreVectorSlice = vector.slice([SliceRange(1..<3)])
    let matrixSlice = matrix.slice(rows: SliceRange(0..<2), columns: SliceRange(1..<3))
    let coreMatrixSlice = matrix.slice([SliceRange(0..<2), SliceRange(1..<3)])
    
    #expect(vectorSlice.offset == coreVectorSlice.offset)
    #expect(vectorSlice.shape == coreVectorSlice.shape)
    #expect(vectorSlice.strides == coreVectorSlice.strides)
    #expect(matrixSlice.offset == coreMatrixSlice.offset)
    #expect(matrixSlice.shape == coreMatrixSlice.shape)
    #expect(matrixSlice.strides == coreMatrixSlice.strides)
}

@Test("Contiguous and non-contiguous views", arguments: [
    ([1.0, 2.0, 3.0], 0, [3], [1], true),
    ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, [2], [1], true),
    ([1.0, 2.0, 3.0, 4.0, 5.0], 0, [3], [2], false),
    ([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1, [3], [2], false)
])
func TensorFlatView_contiguity(elements: [Double], offset: Int, shape: [Int], strides: [Int], expected: Bool) {
    let storage = TensorStorage(elements)
    let view = TensorFlatView(storage: storage, offset: offset, shape: shape, strides: strides)
    
    #expect(view.isContiguous == expected)
}

@Test func TensorFlatView_scalarViewIsContiguous() {
    let view = TensorFlatView<Double>(shape: [], elements: [42.0])
    
    #expect(view.isContiguous)
}

@Test func TensorFlatView_emptyDimensionViewIsContiguousWithColumnMajorStrides() {
    let view = TensorFlatView<Double>(shape: [0, 3])
    
    #expect(view.strides == [1, 0])
    #expect(view.isContiguous)
}

@Test func TensorFlatView_readOnlyAccessDoesNotCopyStorage() {
    let storage = TensorStorage([1.0, 2.0, 3.0])
    let view = TensorFlatView(storage: storage, offset: 0, shape: [3], strides: [1])
    
    _ = view[[1]]
    
    #expect(view.storage === storage)
}
