public enum BLAS {
    case accelerate
    case openBLAS
    
    public static var `default`: BLAS {
        #if canImport(Accelerate)
        return .accelerate
        #else
        return .openBLAS
        #endif
    }
}
