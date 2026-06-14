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

    func value(_ i: [Int]) -> S {
        var result = S.zero
        for term in terms {
            result += term.coefficient * term.view.storage.elements[term.view.storageIndex(forUncheckedIndex: i)]
        }
        return result
    }

    func materializedElements() -> [S] {
        var elements: [S] = []
        let count = shape.reduce(1, *)
        elements.reserveCapacity(count)
        var i = Array(repeating: 0, count: shape.count)
        for _ in 0..<count {
            elements.append(value(i))
            Self.increment(&i, shape: shape)
        }
        return elements
    }

    func assign(to destination: inout TensorFlatView<S>) {
        precondition(destination.shape == shape, "Assigned slice shape \(shape) must match destination slice shape")
        var i = Array(repeating: 0, count: shape.count)
        for _ in 0..<shape.reduce(1, *) {
            destination.storage.elements[destination.storageIndex(forUncheckedIndex: i)] = value(i)
            Self.increment(&i, shape: shape)
        }
    }

    private static func increment(_ i: inout [Int], shape: [Int]) {
        for dimension in 0..<shape.count {
            i[dimension] += 1
            if i[dimension] < shape[dimension] { return }
            i[dimension] = 0
        }
    }
}
