import Numerics

public enum BLASComplexStorage {
    public static func interleaved<RealType: Real>(_ values: [Numerics.Complex<RealType>]) -> [RealType] {
        values.flatMap { [$0.real, $0.imaginary] }
    }

    public static func complexValues<RealType: Real>(_ values: [RealType]) -> [Numerics.Complex<RealType>] {
        stride(from: 0, to: values.count, by: 2).map { Numerics.Complex(values[$0], values[$0 + 1]) }
    }
}
