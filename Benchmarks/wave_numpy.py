import argparse
import time

import numpy as np


def summarize(times):
    ordered = sorted(times)
    return ordered[0] * 1000.0, ordered[len(ordered) // 2] * 1000.0


def measure(samples, iterations, body):
    checksum = body()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        sample_checksum = 0.0
        for _ in range(iterations):
            sample_checksum += body()
        end = time.perf_counter()
        checksum += sample_checksum
        times.append((end - start) / iterations)
    return summarize(times), checksum


class NumPyWave:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.u = np.zeros((3, nx, ny))
        self.alpha = np.zeros((nx, ny))
        self.alpha[:, :] = 0.25
        self.pixels = np.zeros((nx, ny, 3), dtype=np.uint8)
        self.u[0, nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 120.0

    def step(self):
        nx = self.nx
        ny = self.ny
        u = self.u
        alpha = self.alpha
        u[2] = u[1]
        u[1] = u[0]
        u[0, 1:nx - 1, 1:ny - 1] = alpha[1:nx - 1, 1:ny - 1] * (
            u[1, 0:nx - 2, 1:ny - 1]
            + u[1, 2:nx, 1:ny - 1]
            + u[1, 1:nx - 1, 0:ny - 2]
            + u[1, 1:nx - 1, 2:ny]
            - 4 * u[1, 1:nx - 1, 1:ny - 1]
        ) + 2 * u[1, 1:nx - 1, 1:ny - 1] - u[2, 1:nx - 1, 1:ny - 1]
        u[0, 1:nx - 1, 1:ny - 1] *= 0.995
        return float(u[0, 1, 1] + u[0, nx - 2, ny - 2])

    def render_pixels(self):
        nx = self.nx
        ny = self.ny
        u = self.u
        pixels = self.pixels
        pixels[1:nx, 1:ny, 0] = np.clip(u[0, 1:nx, 1:ny] + 128, 0, 255)
        pixels[1:nx, 1:ny, 1] = np.clip(u[1, 1:nx, 1:ny] + 128, 0, 255)
        pixels[1:nx, 1:ny, 2] = np.clip(u[2, 1:nx, 1:ny] + 128, 0, 255)
        return float(int(pixels[1, 1, 0]) + int(pixels[nx - 1, ny - 1, 2]))


def print_result(name, result):
    (best, median), checksum = result
    print(f"  {name}")
    print(f"    median {median:.4f} ms, best {best:.4f} ms")
    print(f"    checksum {checksum:.4f}")
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=300)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    print("NumPy wave stencil")
    print(f"shape: {args.nx}x{args.ny}")
    print("")
    print_result("update", measure(args.samples, args.iterations, NumPyWave(args.nx, args.ny).step))
    print_result("pixel fill", measure(args.samples, args.iterations, NumPyWave(args.nx, args.ny).render_pixels))
    print("NumPy matrix slice primitives")
    print("")
    matrix = np.arange(args.nx * args.ny, dtype=float).reshape((args.nx, args.ny)) % 17
    r = slice(1, args.nx - 1)
    c = slice(1, args.ny - 1)
    destination = matrix.copy()
    print_result("slice view", measure(args.samples, args.iterations, lambda: float(matrix[r, c][0, 0])))
    print_result(
        "slice add",
        measure(args.samples, args.iterations, lambda: float((matrix[1:args.nx - 1, 0:args.ny - 2]
                                                             + matrix[1:args.nx - 1, 2:args.ny])[0, 0])),
    )
    print_result(
        "slice scalar multiply",
        measure(args.samples, args.iterations, lambda: float((4 * matrix[r, c])[0, 0])),
    )

    def assign_slice():
        destination[r, c] = matrix[r, c]
        return float(destination[1, 1])

    print_result("slice assign", measure(args.samples, args.iterations, assign_slice))


if __name__ == "__main__":
    main()
