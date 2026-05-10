import Testing
@testable import Tensors

@Test func TensorStorage_wrapsElements() {
    let storage = TensorStorage([1.0, 2.0, 3.0])
    
    #expect(storage.elements == [1.0, 2.0, 3.0])
    #expect(storage[1] == 2.0)
}

@Test func TensorStorage_isSharedByReference() {
    let storage = TensorStorage([1.0, 2.0, 3.0])
    let alias = storage
    
    alias[1] = 99.0
    
    #expect(storage === alias)
    #expect(storage.elements == [1.0, 99.0, 3.0])
    #expect(storage[1] == 99.0)
}
