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
    
    // MARK: - PluTensor conformance  
    public func approximatelyEquals(_ other: Self, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
        return self.isApproximatelyEqual(to: other, relativeTolerance: tolerance)
    }
}

extension Complex: PluTensor, PluScalar {
    // MARK: - PluScalar conformance
    public func round() -> Complex {
        return Complex(real.round(), imaginary.round())
    }
    
    // MARK: - PluTensor conformance
    public func approximatelyEquals(_ other: Complex, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
        return self.isApproximatelyEqual(to: other, relativeTolerance: tolerance)
    }
}
