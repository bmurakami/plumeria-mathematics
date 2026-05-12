import Testing
@testable import Tensors

@Test func VectorFlatView_wrapsRankOneTensorFlatView() {
    let vector = VectorFlatView<Double>([1.0, 2.0, 3.0])
    
    #expect(vector.size == 3)
    #expect(vector.shape == [3])
    #expect(vector.rank == 1)
    #expect(vector.elements == [1.0, 2.0, 3.0])
    #expect(vector.isContiguous)
}

@Test func VectorFlatView_readsAndWritesElements() {
    var vector = VectorFlatView<Double>([1.0, 2.0, 3.0])
    
    vector[1] = 99.0
    
    #expect(vector[1] == 99.0)
    #expect(vector.elements == [1.0, 99.0, 3.0])
}

@Test func VectorFlatView_slicesWithoutCopying() {
    let vector = VectorFlatView<Double>([1.0, 2.0, 3.0, 4.0, 5.0])
    let slice = vector.slice(SliceRange(1..<5, step: 2))
    
    #expect(slice.view.storage === vector.view.storage)
    #expect(slice.shape == [2])
    #expect(slice.elements == [2.0, 4.0])
    #expect(!slice.isContiguous)
}

@Test func VectorFlatView_mutatingSliceCopiesOnWrite() {
    let vector = VectorFlatView<Double>([1.0, 2.0, 3.0])
    var slice = vector.slice(SliceRange(1..<3))
    
    slice[0] = 99.0
    
    #expect(slice.view.storage !== vector.view.storage)
    #expect(vector.elements == [1.0, 2.0, 3.0])
    #expect(slice.elements == [99.0, 3.0])
}
