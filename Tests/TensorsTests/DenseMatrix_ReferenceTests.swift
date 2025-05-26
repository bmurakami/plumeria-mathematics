//import Testing
//@testable import Tensors
//
//@Test func DenseMatrix_Reference_initializerWithValues() {
//    var m = DenseMatrix_Reference.init([[1.0, 2.0], [3.0, 4.0]])
//    #expect(m[0, 0] == 1.0)
//    #expect(m[1, 0] == 3.0)
//    #expect(m[0, 1] == 2.0)
//    #expect(m[1, 1] == 4.0)
//    
//    m[1, 0] = 3.14
//    #expect(m[1, 0] == 3.14)
//}
//
//@Test func DenseMatrix_Reference_initializerWithRowsAndColumns() {
//    var m = DenseMatrix_Reference.init(rows: 2, columns: 3)
//    
//    #expect(m.rows == 2)
//    #expect(m.columns == 3)
//    
//    for i in 0..<m.rows {
//        for j in 0..<m.columns {
//            #expect(m[i, j] == 0)
//        }
//    }
//    
//    m[1, 2] = 3.14
//    #expect(m[1, 2] == 3.14)
//}
//
//@Test func DenseMatrix_Reference_vectorMultiplication() {
//    let A = DenseMatrix_Reference.init([[1.0, 2.0],
//                             [3.0, 4.0]])
//    let v = DenseVector([2.0, 3.0])
//    let b = A * v
//
//    #expect(b == DenseVector([8.0, 18.0]))
//}
//
//@Test func DenseMatrix_Reference_transpose() {
//    let m = DenseMatrix_Reference.init([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
//    let mt = m.t
//    for i in 0..<mt.rows {
//        for j in 0..<mt.columns {
//            #expect(mt[i, j] == m[j, i])
//        }
//    }
//}
//
//@Test func DenseMatrix_Reference_flatten() {
//    let m = DenseMatrix_Reference.init([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
//    
//    #expect(m.flatten(columnMajorOrder: true) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//    #expect(m.flatten(columnMajorOrder: false) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
//}
//
//@Test func DenseMatrix_Reference_toArray() {
//    let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
//    let m = DenseMatrix_Reference(a)
//    #expect(m.toArray() == a)
//}
