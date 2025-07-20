import Foundation
import Numerics

public typealias Complex = Numerics.Complex<Double>

public protocol PluScalar: ElementaryFunctions & PluTensor & Numeric {
    func round() -> Self
}

extension Double: PluScalar {
    // MARK: - PluScalar conformance
    public func round() -> Double {
        let precision = 14
        let multiplier = Foundation.pow(10.0, Double(precision))
        return (self * multiplier).rounded() / multiplier
    }
}

extension Complex: PluTensor, PluScalar {
    // MARK: - PluScalar conformance
    public func round() -> Complex {
        return Complex(real.round(), imaginary.round())
    }
}
