protocol ApproximatelyEquatable {
    func approximatelyEquals(_ other: Self) -> Bool
}

extension FloatingPoint where Self: ApproximatelyEquatable {
    func approximatelyEquals(_ other: Self) -> Bool {
        let tolerance: Self = Self.ulpOfOne * Self(10)
        return abs(self - other) < tolerance * (abs(self) + abs(other))
    }
}

extension Float: ApproximatelyEquatable {}
extension Double: ApproximatelyEquatable {}
