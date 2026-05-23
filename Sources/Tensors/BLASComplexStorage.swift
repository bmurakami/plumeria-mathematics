import Numerics

public enum BLASComplexStorage {
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
