public protocol Tensor {
    func approximatelyEquals(_ other: Self, tolerance: Self) -> Bool
}
