public struct SliceRange: Equatable, Sendable {
    public let start: Int
    public let length: Int
    public let step: Int
    
    public init(_ range: Range<Int>, step: Int = 1) {
        precondition(range.lowerBound >= 0, "Slice start must be non-negative")
        precondition(step > 0, "Slice step must be positive")
                
        self.start = range.lowerBound
        self.length = range.isEmpty ? 0 : ((range.count - 1) / step) + 1
        self.step = step
    }
    
    public static func all(length: Int) -> SliceRange {
        precondition(length >= 0, "Slice length must be non-negative")
        
        return SliceRange(0..<length)
    }
}
