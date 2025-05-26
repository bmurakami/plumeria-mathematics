public protocol Scalar: Tensor {
    static var zero: Self { get }

//    func approximatelyEquals(_ other: Self, tolerance: Double) -> Bool
}

extension Double: Scalar {
    public func approximatelyEquals(_ other: Self, tolerance: Double = 10 * Double.ulpOfOne) -> Bool {
        return abs(self - other) < tolerance * (abs(self) + abs(other))
    }
}

public struct Complex: Scalar {
    private let x: Double
    private let y: Double
    
    public var re:  Double { x }
    public var im:  Double { y }
    public static var zero: Complex { Complex(0.0, 0.0) }

    public init(_ real: Double, _ imaginary: Double) {
        self.x = real
        self.y = imaginary
    }
    
    // MARK: - Tensor conformance
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
