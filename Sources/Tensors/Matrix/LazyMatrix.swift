struct LazyMatrix<S: PluScalar> {
    struct Term {
        let coefficient: S
        let view: TensorFlatView<S>
    }

    let terms: [Term]

    init(terms: [Term]) {
        self.terms = terms
    }

    static func view(_ view: TensorFlatView<S>) -> LazyMatrix<S> {
        LazyMatrix(terms: [Term(coefficient: 1, view: view)])
    }

    func adding(_ other: LazyMatrix<S>) -> LazyMatrix<S> {
        LazyMatrix(terms: terms + other.terms)
    }

    func subtracting(_ other: LazyMatrix<S>) -> LazyMatrix<S> {
        LazyMatrix(terms: terms + other.terms.map { Term(coefficient: S.zero - $0.coefficient, view: $0.view) })
    }

    func scaled(by scalar: S) -> LazyMatrix<S> {
        LazyMatrix(terms: terms.map { Term(coefficient: scalar * $0.coefficient, view: $0.view) })
    }

    func value(row: Int, column: Int) -> S {
        var result = S.zero
        for term in terms {
            let index = term.view.offset + row * term.view.strides[0] + column * term.view.strides[1]
            result += term.coefficient * term.view.storage.elements[index]
        }
        return result
    }

    func materializedElements(rows: Int, columns: Int) -> [S] {
        var elements = Array(repeating: S.zero, count: rows * columns)
        for column in 0..<columns {
            let resultColumn = column * rows
            for row in 0..<rows { elements[resultColumn + row] = value(row: row, column: column) }
        }
        return elements
    }

    func assign(to destination: inout TensorFlatView<S>) {
        for column in 0..<destination.shape[1] {
            let destinationColumn = destination.offset + column * destination.strides[1]
            for row in 0..<destination.shape[0] {
                destination.storage[destinationColumn + row * destination.strides[0]] = value(row: row, column: column)
            }
        }
    }
}

extension LazyMatrix where S == Double {
    func assign(to destination: inout TensorFlatView<Double>) {
        if terms.count == 7 {
            assignSevenTerms(to: &destination)
            return
        }
        let termData = terms.map {
            (
                coefficient: $0.coefficient, elements: $0.view.storage.elements, offset: $0.view.offset,
                rowStride: $0.view.strides[0], columnStride: $0.view.strides[1]
            )
        }
        for column in 0..<destination.shape[1] {
            let destinationColumn = destination.offset + column * destination.strides[1]
            for row in 0..<destination.shape[0] {
                var result = 0.0
                for term in termData {
                    let index = term.offset + row * term.rowStride + column * term.columnStride
                    result += term.coefficient * term.elements[index]
                }
                destination.storage.elements[destinationColumn + row * destination.strides[0]] = result
            }
        }
    }

    private func assignSevenTerms(to destination: inout TensorFlatView<Double>) {
        let t0 = terms[0], t1 = terms[1], t2 = terms[2], t3 = terms[3], t4 = terms[4], t5 = terms[5], t6 = terms[6]
        let e0 = t0.view.storage.elements, e1 = t1.view.storage.elements, e2 = t2.view.storage.elements
        let e3 = t3.view.storage.elements, e4 = t4.view.storage.elements, e5 = t5.view.storage.elements
        let e6 = t6.view.storage.elements
        if destination.strides[0] == 1 && terms.allSatisfy({ $0.view.strides[0] == 1 }) {
            assignSevenRowContiguousTerms(to: &destination)
            return
        }
        let a0 = t0.coefficient, a1 = t1.coefficient, a2 = t2.coefficient, a3 = t3.coefficient
        let a4 = t4.coefficient, a5 = t5.coefficient, a6 = t6.coefficient
        let s0 = t0.view.strides[0], s1 = t1.view.strides[0], s2 = t2.view.strides[0], s3 = t3.view.strides[0]
        let s4 = t4.view.strides[0], s5 = t5.view.strides[0], s6 = t6.view.strides[0]
        let ds = destination.strides[0]
        for column in 0..<destination.shape[1] {
            let d = destination.offset + column * destination.strides[1]
            let c0 = t0.view.offset + column * t0.view.strides[1]
            let c1 = t1.view.offset + column * t1.view.strides[1]
            let c2 = t2.view.offset + column * t2.view.strides[1]
            let c3 = t3.view.offset + column * t3.view.strides[1]
            let c4 = t4.view.offset + column * t4.view.strides[1]
            let c5 = t5.view.offset + column * t5.view.strides[1]
            let c6 = t6.view.offset + column * t6.view.strides[1]
            for row in 0..<destination.shape[0] {
                var result = a0 * e0[c0 + row * s0]
                result += a1 * e1[c1 + row * s1]
                result += a2 * e2[c2 + row * s2]
                result += a3 * e3[c3 + row * s3]
                result += a4 * e4[c4 + row * s4]
                result += a5 * e5[c5 + row * s5]
                result += a6 * e6[c6 + row * s6]
                destination.storage.elements[d + row * ds] = result
            }
        }
    }

    private func assignSevenRowContiguousTerms(to destination: inout TensorFlatView<Double>) {
        let t0 = terms[0], t1 = terms[1], t2 = terms[2], t3 = terms[3], t4 = terms[4], t5 = terms[5], t6 = terms[6]
        if t4.view.sameStorageRegion(as: t5.view) {
            assignSevenRowContiguousTermsWithCombinedCenter(to: &destination)
            return
        }
        let e0 = t0.view.storage.elements, e1 = t1.view.storage.elements, e2 = t2.view.storage.elements
        let e3 = t3.view.storage.elements, e4 = t4.view.storage.elements, e5 = t5.view.storage.elements
        let e6 = t6.view.storage.elements
        let a0 = t0.coefficient, a1 = t1.coefficient, a2 = t2.coefficient, a3 = t3.coefficient
        let a4 = t4.coefficient, a5 = t5.coefficient, a6 = t6.coefficient
        for column in 0..<destination.shape[1] {
            var d = destination.offset + column * destination.strides[1]
            var i0 = t0.view.offset + column * t0.view.strides[1]
            var i1 = t1.view.offset + column * t1.view.strides[1]
            var i2 = t2.view.offset + column * t2.view.strides[1]
            var i3 = t3.view.offset + column * t3.view.strides[1]
            var i4 = t4.view.offset + column * t4.view.strides[1]
            var i5 = t5.view.offset + column * t5.view.strides[1]
            var i6 = t6.view.offset + column * t6.view.strides[1]
            for _ in 0..<destination.shape[0] {
                var result = a0 * e0[i0]
                result += a1 * e1[i1]
                result += a2 * e2[i2]
                result += a3 * e3[i3]
                result += a4 * e4[i4]
                result += a5 * e5[i5]
                result += a6 * e6[i6]
                destination.storage.elements[d] = result
                d += 1
                i0 += 1
                i1 += 1
                i2 += 1
                i3 += 1
                i4 += 1
                i5 += 1
                i6 += 1
            }
        }
    }

    private func assignSevenRowContiguousTermsWithCombinedCenter(to destination: inout TensorFlatView<Double>) {
        let t0 = terms[0], t1 = terms[1], t2 = terms[2], t3 = terms[3], t4 = terms[4], t6 = terms[6]
        if t0.view.storage === t1.view.storage && t0.view.storage === t2.view.storage
            && t0.view.storage === t3.view.storage && t0.view.storage === t4.view.storage {
            assignSevenRowContiguousTermsWithTwoStorages(to: &destination)
            return
        }
        let e0 = t0.view.storage.elements, e1 = t1.view.storage.elements, e2 = t2.view.storage.elements
        let e3 = t3.view.storage.elements, e4 = t4.view.storage.elements, e6 = t6.view.storage.elements
        let a0 = t0.coefficient, a1 = t1.coefficient, a2 = t2.coefficient, a3 = t3.coefficient
        let a4 = terms[4].coefficient + terms[5].coefficient, a6 = t6.coefficient
        for column in 0..<destination.shape[1] {
            var d = destination.offset + column * destination.strides[1]
            var i0 = t0.view.offset + column * t0.view.strides[1]
            var i1 = t1.view.offset + column * t1.view.strides[1]
            var i2 = t2.view.offset + column * t2.view.strides[1]
            var i3 = t3.view.offset + column * t3.view.strides[1]
            var i4 = t4.view.offset + column * t4.view.strides[1]
            var i6 = t6.view.offset + column * t6.view.strides[1]
            for _ in 0..<destination.shape[0] {
                var result = a0 * e0[i0]
                result += a1 * e1[i1]
                result += a2 * e2[i2]
                result += a3 * e3[i3]
                result += a4 * e4[i4]
                result += a6 * e6[i6]
                destination.storage.elements[d] = result
                d += 1
                i0 += 1
                i1 += 1
                i2 += 1
                i3 += 1
                i4 += 1
                i6 += 1
            }
        }
    }

    private func assignSevenRowContiguousTermsWithTwoStorages(to destination: inout TensorFlatView<Double>) {
        let t0 = terms[0], t1 = terms[1], t2 = terms[2], t3 = terms[3], t4 = terms[4], t6 = terms[6]
        if destination.storage !== t0.view.storage && destination.storage !== t6.view.storage {
            assignDistinctSevenRowContiguousTermsWithTwoStorages(to: &destination)
            return
        }
        let source = t0.view.storage.elements, older = t6.view.storage.elements
        let a0 = t0.coefficient, a1 = t1.coefficient, a2 = t2.coefficient, a3 = t3.coefficient
        let a4 = terms[4].coefficient + terms[5].coefficient, a6 = t6.coefficient
        for column in 0..<destination.shape[1] {
            var d = destination.offset + column * destination.strides[1]
            var i0 = t0.view.offset + column * t0.view.strides[1]
            var i1 = t1.view.offset + column * t1.view.strides[1]
            var i2 = t2.view.offset + column * t2.view.strides[1]
            var i3 = t3.view.offset + column * t3.view.strides[1]
            var i4 = t4.view.offset + column * t4.view.strides[1]
            var i6 = t6.view.offset + column * t6.view.strides[1]
            for _ in 0..<destination.shape[0] {
                var result = a0 * source[i0]
                result += a1 * source[i1]
                result += a2 * source[i2]
                result += a3 * source[i3]
                result += a4 * source[i4]
                result += a6 * older[i6]
                destination.storage.elements[d] = result
                d += 1
                i0 += 1
                i1 += 1
                i2 += 1
                i3 += 1
                i4 += 1
                i6 += 1
            }
        }
    }

    private func assignDistinctSevenRowContiguousTermsWithTwoStorages(to destination: inout TensorFlatView<Double>) {
        let t0 = terms[0], t1 = terms[1], t2 = terms[2], t3 = terms[3], t4 = terms[4], t6 = terms[6]
        let a0 = t0.coefficient, a1 = t1.coefficient, a2 = t2.coefficient, a3 = t3.coefficient
        let a4 = terms[4].coefficient + terms[5].coefficient, a6 = t6.coefficient
        destination.storage.elements.withUnsafeMutableBufferPointer { destinationElements in
            t0.view.storage.elements.withUnsafeBufferPointer { source in
                t6.view.storage.elements.withUnsafeBufferPointer { older in
                    for column in 0..<destination.shape[1] {
                        var d = destination.offset + column * destination.strides[1]
                        var i0 = t0.view.offset + column * t0.view.strides[1]
                        var i1 = t1.view.offset + column * t1.view.strides[1]
                        var i2 = t2.view.offset + column * t2.view.strides[1]
                        var i3 = t3.view.offset + column * t3.view.strides[1]
                        var i4 = t4.view.offset + column * t4.view.strides[1]
                        var i6 = t6.view.offset + column * t6.view.strides[1]
                        for _ in 0..<destination.shape[0] {
                            var result = a0 * source[i0]
                            result += a1 * source[i1]
                            result += a2 * source[i2]
                            result += a3 * source[i3]
                            result += a4 * source[i4]
                            result += a6 * older[i6]
                            destinationElements[d] = result
                            d += 1
                            i0 += 1
                            i1 += 1
                            i2 += 1
                            i3 += 1
                            i4 += 1
                            i6 += 1
                        }
                    }
                }
            }
        }
    }
}

extension TensorFlatView {
    fileprivate func sameStorageRegion(as other: TensorFlatView<Scalar>) -> Bool {
        storage === other.storage && offset == other.offset && shape == other.shape && strides == other.strides
    }
}
