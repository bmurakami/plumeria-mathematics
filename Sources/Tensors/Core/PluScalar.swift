import Foundation
import Numerics

public typealias ComplexDouble = Numerics.Complex<Double>
public typealias ComplexFloat = Numerics.Complex<Float>

public protocol PluScalar: ElementaryFunctions & PluTensor & Numeric where Magnitude: FloatingPoint {
    static func / (lhs: Self, rhs: Self) -> Self
    func round(precision: Int) -> Self
}

public protocol ComplexScalar: PluScalar {
    var star: Self { get }
    var mod: Magnitude { get }
    var arg: Magnitude { get }
}

extension Double: PluScalar {
    public typealias S = Double

    public func round(precision: Int = 14) -> Double {
        let multiplier = Foundation.pow(10.0, Double(precision))
        return (self * multiplier).rounded() / multiplier
    }
}

extension Float: PluScalar {
    public typealias S = Float

    public func round(precision: Int = 6) -> Float {
        let multiplier = Float(Foundation.pow(10.0, Double(precision)))
        return (self * multiplier).rounded() / multiplier
    }
}

extension Numerics.Complex: TensorArithmetic, PluTensor, PluScalar {
    public typealias S = Numerics.Complex<RealType>

    public static var i: Self { Self(RealType(0), RealType(1)) }

    public func round(precision: Int) -> Self {
        let multiplier = RealType.pow(RealType(10), RealType(precision))
        return Self((real * multiplier).rounded() / multiplier,
                    (imaginary * multiplier).rounded() / multiplier)
    }
}

extension Numerics.Complex: ComplexScalar {
    public var star: Self { conjugate }
    public var mod: RealType { length }
    public var arg: RealType { phase }
}

public func + (left: Double, right: ComplexDouble) -> ComplexDouble { ComplexDouble(left, 0.0) + right }
public func + (left: ComplexDouble, right: Double) -> ComplexDouble { left + ComplexDouble(right, 0.0) }
public func - (left: Double, right: ComplexDouble) -> ComplexDouble { ComplexDouble(left, 0.0) - right }
public func - (left: ComplexDouble, right: Double) -> ComplexDouble { left - ComplexDouble(right, 0.0) }
public func * (left: Double, right: ComplexDouble) -> ComplexDouble { ComplexDouble(left, 0.0) * right }
public func * (left: ComplexDouble, right: Double) -> ComplexDouble { left * ComplexDouble(right, 0.0) }
public func / (left: Double, right: ComplexDouble) -> ComplexDouble { ComplexDouble(left, 0.0) / right }
public func / (left: ComplexDouble, right: Double) -> ComplexDouble { left / ComplexDouble(right, 0.0) }

public func + (left: Float, right: ComplexFloat) -> ComplexFloat { ComplexFloat(left, 0.0) + right }
public func + (left: ComplexFloat, right: Float) -> ComplexFloat { left + ComplexFloat(right, 0.0) }
public func - (left: Float, right: ComplexFloat) -> ComplexFloat { ComplexFloat(left, 0.0) - right }
public func - (left: ComplexFloat, right: Float) -> ComplexFloat { left - ComplexFloat(right, 0.0) }
public func * (left: Float, right: ComplexFloat) -> ComplexFloat { ComplexFloat(left, 0.0) * right }
public func * (left: ComplexFloat, right: Float) -> ComplexFloat { left * ComplexFloat(right, 0.0) }
public func / (left: Float, right: ComplexFloat) -> ComplexFloat { ComplexFloat(left, 0.0) / right }
public func / (left: ComplexFloat, right: Float) -> ComplexFloat { left / ComplexFloat(right, 0.0) }

extension PluScalar {
    public func round() -> Self {
        switch Self.self {
        case is Float.Type, is ComplexFloat.Type:
            return round(precision: 6)
        default:
            return round(precision: 14)
        }
    }
}
