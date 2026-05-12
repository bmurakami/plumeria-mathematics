import Testing
@testable import Tensors

@Test func TensorSliceIndex_rangeNormalizesToSliceRange() {
    let range = TensorSliceIndex.range(1..<4).sliceRange(dimensionSize: 5)
    
    #expect(range == SliceRange(1..<4))
}

@Test func TensorSliceIndex_rangeHelperCreatesRangeIndex() {
    let range = range(1..<4).sliceRange(dimensionSize: 5)
    
    #expect(range == SliceRange(1..<4))
}

@Test func TensorSliceIndex_stepNormalizesToSliceRange() {
    let range = step(0..<5, by: 2).sliceRange(dimensionSize: 5)
    
    #expect(range == SliceRange(0..<5, step: 2))
}

@Test func TensorSliceIndex_allNormalizesToFullDimension() {
    let range = all.sliceRange(dimensionSize: 4)
    
    #expect(range == SliceRange(0..<4))
}

@Test func TensorSliceIndex_indexNormalizesToFixedIndex() {
    let index = TensorSliceIndex.index(2).fixedIndex(dimensionSize: 4)
    
    #expect(index == 2)
}
