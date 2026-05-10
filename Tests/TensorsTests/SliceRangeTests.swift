import Testing
@testable import Tensors

@Test func SliceRange_defaultsToUnitStep() {
    let range = SliceRange(2..<7)
    
    #expect(range.start == 2)
    #expect(range.length == 5)
    #expect(range.step == 1)
}

@Test func SliceRange_initializesFromSwiftRangeWithStep() {
    let range = SliceRange(2..<8, step: 2)
    
    #expect(range.start == 2)
    #expect(range.length == 3)
    #expect(range.step == 2)
}

@Test func SliceRange_initializesFromEmptySwiftRangeWithStep() {
    let range = SliceRange(2..<2, step: 2)
    
    #expect(range.start == 2)
    #expect(range.length == 0)
    #expect(range.step == 2)
}

@Test func SliceRange_initializesFromSwiftRangeWithUnevenStep() {
    let range = SliceRange(0..<9, step: 2)
    
    #expect(range.start == 0)
    #expect(range.length == 5)
    #expect(range.step == 2)
}

@Test func SliceRange_allCoversLengthFromZero() {
    let range = SliceRange.all(length: 4)
    
    #expect(range.start == 0)
    #expect(range.length == 4)
    #expect(range.step == 1)
}

@Test func SliceRange_allAllowsEmptyLength() {
    let range = SliceRange.all(length: 0)
    
    #expect(range.start == 0)
    #expect(range.length == 0)
    #expect(range.step == 1)
}
