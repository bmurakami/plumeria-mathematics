import Testing
@testable import Tensors

@Test func approximatelyEquals_defaultTolerance() {
    let testCases = [
        (1e-16, true),
        (1e-15, true),
        (1e-14, false),
        (1e-13, false)
    ]
    let x = 1.0
    for testCase in testCases {
        let epsilon = testCase.0
        let y = 1.0 + epsilon
        #expect(x.approximatelyEquals(y) == testCase.1)
    }
}

@Test func approximatelyEquals_tolerance() {
    let testCases = [
        (1e-5, true),
        (1e-4, true),
        (1e-3, false),
        (1e-2, false)
    ]
    let x = 1.0
    for testCase in testCases {
        let epsilon = testCase.0
        let y = 1.0 + epsilon
        #expect(x.approximatelyEquals(y, tolerance: 1e-4) == testCase.1)
    }
}
