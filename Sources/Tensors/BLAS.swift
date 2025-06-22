public enum BLAS {
    #if canImport(Accelerate)
    case accelerate
    #endif
    case openBLAS
    
    public static var `default`: BLAS {
        #if canImport(Accelerate)
        return .accelerate
        #else
        return .openBLAS
        #endif
    }
}
