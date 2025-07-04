import Foundation

public protocol PluScalar: PluTensor {
    static var zero: Self { get }
    
    func round() -> Self
    static func * (lhs: Self, rhs: Self) -> Self
}

extension Double: PluScalar {
    // MARK: - PluScalar conformance
    public func round() -> Double {
        let precision = 14
        let multiplier = pow(10.0, Double(precision))
        return (self * multiplier).rounded() / multiplier
    }
    
    // MARK: - PluTensor conformance
    public func approximatelyEquals(_ other: Self, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
        return abs(self - other) < tolerance * (abs(self) + abs(other))
    }
}

public struct Complex: PluScalar {
    private let x: Double
    private let y: Double
    
    public var re:  Double { x }
    public var im:  Double { y }

    public init(_ real: Double, _ imaginary: Double) {
        self.x = real
        self.y = imaginary
    }
    
    // MARK: - PluScalar conformance
    public static var zero: Complex { Complex(0.0, 0.0) }
    
    public func round() -> Complex {
        return Complex(x.round(), y.round())
    }
    
    public static func * (lhs: Complex, rhs: Complex) -> Complex {
        return Complex(lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x)
    }

    // MARK: - PluTensor conformance
    public static func + (lhs: Complex, rhs: Complex) -> Complex {
        return Complex(lhs.x + rhs.x, lhs.y + rhs.y)
    }

    public static prefix func - (lhs: Complex) -> Complex {
        return Complex(-lhs.x, -lhs.y)
    }
    
    public func approximatelyEquals(_ other: Complex, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
        return re.approximatelyEquals(other.re) && im.approximatelyEquals(other.im)
    }
}

