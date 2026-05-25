struct LazyTensor<S: PluScalar> {
    struct Term {
        let coefficient: S
        let view: TensorFlatView<S>
    }

    let terms: [Term]
    var shape: [Int] { terms[0].view.shape }

    init(terms: [Term]) {
        precondition(!terms.isEmpty, "Lazy tensor must contain at least one term")
        let shape = terms[0].view.shape
        precondition(terms.allSatisfy { $0.view.shape == shape }, "Lazy tensor terms must have the same shape")
        self.terms = terms
    }

    static func view(_ view: TensorFlatView<S>) -> LazyTensor<S> {
        LazyTensor(terms: [Term(coefficient: 1, view: view)])
    }

    func adding(_ other: LazyTensor<S>) -> LazyTensor<S> {
        precondition(shape == other.shape, "Tensors must have the same shape")
        return LazyTensor(terms: terms + other.terms)
    }

    func subtracting(_ other: LazyTensor<S>) -> LazyTensor<S> {
        precondition(shape == other.shape, "Tensors must have the same shape")
        return LazyTensor(terms: terms + other.terms.map { Term(coefficient: S.zero - $0.coefficient, view: $0.view) })
    }

    func scaled(by scalar: S) -> LazyTensor<S> {
        LazyTensor(terms: terms.map { Term(coefficient: scalar * $0.coefficient, view: $0.view) })
    }

    func value(_ index: [Int]) -> S {
        var result = S.zero
        for term in terms {
            result += term.coefficient * term.view.storage.elements[term.view.storageIndex(forUncheckedIndex: index)]
        }
        return result
    }

    func materializedElements() -> [S] {
        var elements: [S] = []
        elements.reserveCapacity(shape.reduce(1, *))
        for index in indexCombinations(for: shape) { elements.append(value(index)) }
        return elements
    }

    func assign(to destination: inout TensorFlatView<S>) {
        precondition(destination.shape == shape, "Assigned slice shape \(shape) must match destination slice shape")
        for index in indexCombinations(for: shape) {
            destination.storage.elements[destination.storageIndex(forUncheckedIndex: index)] = value(index)
        }
    }

    private func indexCombinations(for shape: [Int]) -> [[Int]] {
        if shape.isEmpty { return [[]] }
        if shape.contains(0) { return [] }
        return (0..<shape.reduce(1, *)).map { flatIndex in
            var remaining = flatIndex
            return shape.map { dimension in
                let index = remaining % dimension
                remaining /= dimension
                return index
            }
        }
    }
}
