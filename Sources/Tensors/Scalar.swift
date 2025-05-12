public protocol ApproximatelyEquatable {
    func approximatelyEquals(_ other: Self, tolerance: Self) -> Bool
}

extension FloatingPoint where Self: ApproximatelyEquatable {
    public func approximatelyEquals(_ other: Self, tolerance: Self = Self.ulpOfOne * Self(10)) -> Bool {
        return abs(self - other) < tolerance * (abs(self) + abs(other))
    }
}

extension Float: ApproximatelyEquatable {}
extension Double: ApproximatelyEquatable {}
