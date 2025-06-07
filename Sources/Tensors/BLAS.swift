public enum BLAS {
    case accelerate
    case openBLAS
    
    public static var `default`: BLAS {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        return .accelerate
        #else
        return .openBLAS
        #endif
    }
}
