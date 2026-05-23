#!/usr/bin/env python3
import argparse
import math
import platform
import time

import numpy as np


def measure(samples, iterations, body):
    checksum = float(body())
    times = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        sample_checksum = 0.0
        for _ in range(iterations):
            sample_checksum += float(body())
        end = time.perf_counter_ns()
        checksum += sample_checksum
        times.append((end - start) / iterations / 1_000_000.0)
    times.sort()
    return {"best": times[0], "median": times[len(times) // 2], "checksum": checksum}


def format_time(value):
    return f"{value:.4f}"


def print_result(operation, size, result):
    print(f"  {operation} {size}")
    print(f"    median {format_time(result['median'])} ms, best {format_time(result['best'])} ms")
    print(f"    checksum {result['checksum']:.4f}")
    print("")


def vector_values(count, dtype=np.float64):
    values = (np.arange(count, dtype=np.float64) % 97 - 48) / 7.0
    return values.astype(dtype, copy=False)


def vector_complex_values(count, dtype=np.complex128):
    values = vector_values(count)
    return (values - 0.5j * values).astype(dtype, copy=False)


def matrix_values(rows, columns, dtype=np.float64):
    row = np.arange(rows, dtype=np.float64).reshape(rows, 1)
    column = np.arange(columns, dtype=np.float64).reshape(1, columns)
    values = ((row * 31 + column * 17) % 101 - 50) / 11.0
    return values.astype(dtype, copy=False)


def matrix_complex_values(rows, columns, dtype=np.complex128):
    values = matrix_values(rows, columns)
    return (values - values * (1j / 3.0)).astype(dtype, copy=False)


def invertible_matrix_values(size, dtype=np.float64):
    values = matrix_values(size, size, dtype=np.float64)
    values += np.eye(size, dtype=np.float64) * size
    return values.astype(dtype, copy=False)


def tensor_values(shape, dtype=np.float64):
    indices = np.indices(shape, dtype=np.int64)
    weighted = np.zeros(shape, dtype=np.int64)
    for offset, index in enumerate(indices):
        weighted += (offset + 1) * (index + 1)
    return (((weighted % 29) - 14) / 5.0).astype(dtype, copy=False)


def tensor_complex_values(shape, dtype=np.complex128):
    values = tensor_values(shape)
    return (values - 0.5j * values).astype(dtype, copy=False)


def determinant_checksum(value):
    if value == 0.0:
        return 0.0
    sign = -1.0 if value < 0.0 else 1.0
    if math.isinf(value):
        return sign * 308.0
    return sign * math.log10(abs(value))


def benchmark_vectors(samples):
    print("Vector")
    small = vector_values(10_000)
    large = vector_values(100_000)
    print_result("add", "10,000", measure(samples, 20, lambda: (small + small)[0] + (small + small)[-1]))
    print_result("scale", "100,000", measure(samples, 10, lambda: (large * 1.25)[0] + (large * 1.25)[-1]))
    print_result("magnitude", "100,000", measure(samples, 10, lambda: np.linalg.norm(large)))


def benchmark_matrices(samples):
    print("Matrix")
    add = matrix_values(256, 256)
    large_add = matrix_values(1_024, 1_024)
    mv = matrix_values(384, 384)
    large_mv = matrix_values(1_536, 1_536)
    vector = vector_values(384)
    large_vector = vector_values(1_536)
    left = matrix_values(128, 128)
    right = matrix_values(128, 128)
    det = invertible_matrix_values(96)
    print_result("add", "256x256", measure(samples, 5, lambda: (add + add)[0, 0] + (add + add)[-1, -1]))
    print_result("add", "1,024x1,024",
                 measure(3, 1, lambda: (large_add + large_add)[0, 0] + (large_add + large_add)[-1, -1]))
    print_result("matrix-vector multiply", "384x384", measure(samples, 1, lambda: (mv @ vector)[0] + (mv @ vector)[-1]))
    print_result("matrix-vector multiply", "1,536x1,536",
                 measure(3, 1, lambda: (large_mv @ large_vector)[0] + (large_mv @ large_vector)[-1]))
    print_result("matrix-matrix multiply", "128x128",
                 measure(samples, 1, lambda: (left @ right)[0, 0] + (left @ right)[-1, -1]))
    print_result("determinant", "96x96", measure(3, 1, lambda: determinant_checksum(np.linalg.det(det))))
    print_result("inverse", "96x96", measure(3, 1, lambda: np.linalg.inv(det)[0, 0] + np.linalg.inv(det)[-1, -1]))


def benchmark_tensors(samples):
    print("Tensor")
    add = tensor_values((40, 40, 10))
    left = tensor_values((16, 24, 16))
    right = tensor_values((16, 16))
    large_left = tensor_values((32, 32, 32))
    large_right = tensor_values((32, 32))
    print_result("add", "40x40x10", measure(samples, 3, lambda: (add + add)[0, 0, 0] + (add + add)[-1, -1, -1]))
    print_result("contraction", "16x24x16,16x16",
                 measure(samples, 1, lambda: np.einsum("ijk,kl", left, right)[0, 0, 0]
                         + np.einsum("ijk,kl", left, right)[-1, -1, -1]))
    print_result("contraction", "32x32x32,32x32",
                 measure(3, 1, lambda: np.einsum("ijk,kl", large_left, large_right)[0, 0, 0]
                         + np.einsum("ijk,kl", large_left, large_right)[-1, -1, -1]))


def benchmark_float_scalars(samples):
    print("Float64 vs. Float32")
    double_vector = vector_values(100_000)
    float_vector = vector_values(100_000, np.float32)
    double_matrix = matrix_values(384, 384)
    float_matrix = matrix_values(384, 384, np.float32)
    double_matrix_vector = vector_values(384)
    float_matrix_vector = vector_values(384, np.float32)
    double_left = matrix_values(128, 128)
    double_right = matrix_values(128, 128)
    float_left = matrix_values(128, 128, np.float32)
    float_right = matrix_values(128, 128, np.float32)
    double_tensor = tensor_values((16, 24, 16))
    double_right_tensor = tensor_values((16, 16))
    float_tensor = tensor_values((16, 24, 16), np.float32)
    float_right_tensor = tensor_values((16, 16), np.float32)
    print_result("Float64 vector add", "100,000",
                 measure(samples, 10, lambda: (double_vector + double_vector)[0] + (double_vector + double_vector)[-1]))
    print_result("Float32 vector add", "100,000",
                 measure(samples, 10, lambda: (float_vector + float_vector)[0] + (float_vector + float_vector)[-1]))
    print_result("Float64 magnitude", "100,000", measure(samples, 10, lambda: np.linalg.norm(double_vector)))
    print_result("Float32 magnitude", "100,000", measure(samples, 10, lambda: np.linalg.norm(float_vector)))
    print_result("Float64 matrix-vector multiply", "384x384",
                 measure(samples, 1, lambda: (double_matrix @ double_matrix_vector)[0]
                         + (double_matrix @ double_matrix_vector)[-1]))
    print_result("Float32 matrix-vector multiply", "384x384",
                 measure(samples, 1, lambda: (float_matrix @ float_matrix_vector)[0]
                         + (float_matrix @ float_matrix_vector)[-1]))
    print_result("Float64 matrix-matrix multiply", "128x128",
                 measure(samples, 1, lambda: (double_left @ double_right)[0, 0] + (double_left @ double_right)[-1, -1]))
    print_result("Float32 matrix-matrix multiply", "128x128",
                 measure(samples, 1, lambda: (float_left @ float_right)[0, 0] + (float_left @ float_right)[-1, -1]))
    print_result("Float64 tensor contraction", "16x24x16,16x16",
                 measure(samples, 1, lambda: np.einsum("ijk,kl", double_tensor, double_right_tensor)[0, 0, 0]))
    print_result("Float32 tensor contraction", "16x24x16,16x16",
                 measure(samples, 1, lambda: np.einsum("ijk,kl", float_tensor, float_right_tensor)[0, 0, 0]))


def benchmark_complex_float_scalars(samples):
    print("Complex128 vs. Complex64")
    complex_vector = vector_complex_values(100_000)
    complex_float_vector = vector_complex_values(100_000, np.complex64)
    complex_matrix = matrix_complex_values(384, 384)
    complex_float_matrix = matrix_complex_values(384, 384, np.complex64)
    complex_matrix_vector = vector_complex_values(384)
    complex_float_matrix_vector = vector_complex_values(384, np.complex64)
    complex_left = matrix_complex_values(128, 128)
    complex_right = matrix_complex_values(128, 128)
    complex_float_left = matrix_complex_values(128, 128, np.complex64)
    complex_float_right = matrix_complex_values(128, 128, np.complex64)
    complex_tensor = tensor_complex_values((16, 24, 16))
    complex_right_tensor = tensor_complex_values((16, 16))
    complex_float_tensor = tensor_complex_values((16, 24, 16), np.complex64)
    complex_float_right_tensor = tensor_complex_values((16, 16), np.complex64)
    print_result("Complex128 vector add", "100,000",
                 measure(samples, 10, lambda: ((complex_vector + complex_vector)[0]
                         + (complex_vector + complex_vector)[-1]).real))
    print_result("Complex64 vector add", "100,000",
                 measure(samples, 10, lambda: ((complex_float_vector + complex_float_vector)[0]
                         + (complex_float_vector + complex_float_vector)[-1]).real))
    print_result("Complex128 magnitude", "100,000", measure(samples, 10, lambda: np.linalg.norm(complex_vector)))
    print_result("Complex64 magnitude", "100,000", measure(samples, 10, lambda: np.linalg.norm(complex_float_vector)))
    print_result("Complex128 matrix-vector multiply", "384x384",
                 measure(samples, 1, lambda: ((complex_matrix @ complex_matrix_vector)[0]
                         + (complex_matrix @ complex_matrix_vector)[-1]).real))
    print_result("Complex64 matrix-vector multiply", "384x384",
                 measure(samples, 1, lambda: ((complex_float_matrix @ complex_float_matrix_vector)[0]
                         + (complex_float_matrix @ complex_float_matrix_vector)[-1]).real))
    print_result("Complex128 matrix-matrix multiply", "128x128",
                 measure(samples, 1, lambda: ((complex_left @ complex_right)[0, 0]
                         + (complex_left @ complex_right)[-1, -1]).real))
    print_result("Complex64 matrix-matrix multiply", "128x128",
                 measure(samples, 1, lambda: ((complex_float_left @ complex_float_right)[0, 0]
                         + (complex_float_left @ complex_float_right)[-1, -1]).real))
    print_result("Complex128 tensor contraction", "16x24x16,16x16",
                 measure(samples, 1, lambda: np.einsum("ijk,kl", complex_tensor, complex_right_tensor)[0, 0, 0].real))
    print_result("Complex64 tensor contraction", "16x24x16,16x16",
                 measure(samples, 1, lambda: np.einsum("ijk,kl", complex_float_tensor,
                                                       complex_float_right_tensor)[0, 0, 0].real))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    print("NumPyBenchmarks")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"Platform: {platform.machine()}")
    print("")
    benchmark_vectors(args.samples)
    benchmark_matrices(args.samples)
    benchmark_tensors(args.samples)
    benchmark_float_scalars(args.samples)
    benchmark_complex_float_scalars(args.samples)


if __name__ == "__main__":
    main()
