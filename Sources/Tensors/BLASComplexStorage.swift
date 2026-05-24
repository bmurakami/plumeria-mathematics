import Numerics

public enum BLASComplexStorage {
    public static func withUnsafeInterleavedStorage<RealType: Real, Result>(
        _ values: [Numerics.Complex<RealType>], _ body: (UnsafeRawPointer) -> Result
    ) -> Result {
        precondition(MemoryLayout<Numerics.Complex<RealType>>.stride == 2 * MemoryLayout<RealType>.stride,
                     "Complex storage must be interleaved real and imaginary values")
        if values.isEmpty {
            var placeholder = [RealType.zero, RealType.zero]
            return placeholder.withUnsafeMutableBytes { body(UnsafeRawPointer($0.baseAddress!)) }
        }
        return values.withUnsafeBytes { body($0.baseAddress!) }
    }

    public static func withUnsafeMutableInterleavedStorage<RealType: Real, Result>(
        _ values: inout [Numerics.Complex<RealType>], _ body: (UnsafeMutableRawPointer) -> Result
    ) -> Result {
        precondition(MemoryLayout<Numerics.Complex<RealType>>.stride == 2 * MemoryLayout<RealType>.stride,
                     "Complex storage must be interleaved real and imaginary values")
        if values.isEmpty {
            var placeholder = [RealType.zero, RealType.zero]
            return placeholder.withUnsafeMutableBytes { body($0.baseAddress!) }
        }
        return values.withUnsafeMutableBytes { body($0.baseAddress!) }
    }

    public static func withUnsafeMutableInterleavedStorage<RealType: Real, Result>(
        _ values: UnsafeMutableBufferPointer<Numerics.Complex<RealType>>,
        _ body: (UnsafeMutableRawPointer) -> Result
    ) -> Result {
        precondition(MemoryLayout<Numerics.Complex<RealType>>.stride == 2 * MemoryLayout<RealType>.stride,
                     "Complex storage must be interleaved real and imaginary values")
        if values.isEmpty {
            var placeholder = [RealType.zero, RealType.zero]
            return placeholder.withUnsafeMutableBytes { body($0.baseAddress!) }
        }
        let bytes = UnsafeMutableRawBufferPointer(values)
        return body(bytes.baseAddress!)
    }

    public static func sum(_ left: [ComplexDouble], _ right: [ComplexDouble]) -> [ComplexDouble] {
        Array<ComplexDouble>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
            left.withUnsafeBytes { left in
                right.withUnsafeBytes { right in
                    let rawResult = UnsafeMutableRawBufferPointer(result)
                    let left = left.bindMemory(to: Double.self), right = right.bindMemory(to: Double.self)
                    let result = rawResult.bindMemory(to: Double.self)
                    for index in 0..<result.count { result[index] = left[index] + right[index] }
                }
            }
            initializedCount = left.count
        }
    }

    public static func sum(_ left: [ComplexFloat], _ right: [ComplexFloat]) -> [ComplexFloat] {
        Array<ComplexFloat>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
            left.withUnsafeBytes { left in
                right.withUnsafeBytes { right in
                    let rawResult = UnsafeMutableRawBufferPointer(result)
                    let left = left.bindMemory(to: Float.self), right = right.bindMemory(to: Float.self)
                    let result = rawResult.bindMemory(to: Float.self)
                    for index in 0..<result.count { result[index] = left[index] + right[index] }
                }
            }
            initializedCount = left.count
        }
    }

    public static func difference(_ left: [ComplexDouble], _ right: [ComplexDouble]) -> [ComplexDouble] {
        Array<ComplexDouble>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
            left.withUnsafeBytes { left in
                right.withUnsafeBytes { right in
                    let rawResult = UnsafeMutableRawBufferPointer(result)
                    let left = left.bindMemory(to: Double.self), right = right.bindMemory(to: Double.self)
                    let result = rawResult.bindMemory(to: Double.self)
                    for index in 0..<result.count { result[index] = left[index] - right[index] }
                }
            }
            initializedCount = left.count
        }
    }

    public static func difference(_ left: [ComplexFloat], _ right: [ComplexFloat]) -> [ComplexFloat] {
        Array<ComplexFloat>(unsafeUninitializedCapacity: left.count) { result, initializedCount in
            left.withUnsafeBytes { left in
                right.withUnsafeBytes { right in
                    let rawResult = UnsafeMutableRawBufferPointer(result)
                    let left = left.bindMemory(to: Float.self), right = right.bindMemory(to: Float.self)
                    let result = rawResult.bindMemory(to: Float.self)
                    for index in 0..<result.count { result[index] = left[index] - right[index] }
                }
            }
            initializedCount = left.count
        }
    }

    public static func interleaved<RealType: Real>(_ values: [Numerics.Complex<RealType>]) -> [RealType] {
        var interleaved = Array(repeating: RealType.zero, count: values.count * 2)
        for index in 0..<values.count {
            interleaved[2 * index] = values[index].real
            interleaved[2 * index + 1] = values[index].imaginary
        }
        return interleaved
    }

    public static func complexValues<RealType: Real>(_ values: [RealType]) -> [Numerics.Complex<RealType>] {
        var complex = Array(repeating: Numerics.Complex<RealType>.zero, count: values.count / 2)
        for index in 0..<complex.count { complex[index] = Numerics.Complex(values[2 * index], values[2 * index + 1]) }
        return complex
    }

}
