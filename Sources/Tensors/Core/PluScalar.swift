import Foundation
import Numerics

public typealias Complex = Numerics.Complex<Double>

public protocol PluScalar: ElementaryFunctions & PluTensor & Numeric {
    static func / (lhs: Self, rhs: Self) -> Self
    func round(precision: Int) -> Self
}

public protocol ComplexScalar: PluScalar {
    var star: Self { get }
    var mod: Magnitude { get }
    var arg: Magnitude { get }
}

extension Double: PluScalar {
    // MARK: - PluScalar conformance
    public typealias S = Double

    public func round(precision: Int = 14) -> Double {
        let multiplier = Foundation.pow(10.0, Double(precision))
        return (self * multiplier).rounded() / multiplier
    }
}

extension Complex: TensorArithmetic, PluTensor, PluScalar, ComplexScalar {
    // MARK: - ComplexScalar conformance
    public typealias S = Complex

    public var star: Complex { conjugate }
    public var mod: Double { length }
    public var arg: Double { phase }

    // MARK: - PluScalar conformance
    public func round(precision: Int = 14) -> Complex {
        return Complex(real.round(precision: precision), imaginary.round(precision: precision))
    }
}

public func + (left: Double, right: Complex) -> Complex {
    Complex(left, 0.0) + right
}

public func + (left: Complex, right: Double) -> Complex {
    left + Complex(right, 0.0)
}

public func - (left: Double, right: Complex) -> Complex {
    Complex(left, 0.0) - right
}

public func - (left: Complex, right: Double) -> Complex {
    left - Complex(right, 0.0)
}

public func * (left: Double, right: Complex) -> Complex {
    Complex(left, 0.0) * right
}

public func * (left: Complex, right: Double) -> Complex {
    left * Complex(right, 0.0)
}

public func / (left: Double, right: Complex) -> Complex {
    Complex(left, 0.0) / right
}

public func / (left: Complex, right: Double) -> Complex {
    left / Complex(right, 0.0)
}

extension PluScalar {
    public func round() -> Self {
        round(precision: 14)
    }
}
