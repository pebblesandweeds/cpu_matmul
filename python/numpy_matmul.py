import time
import numpy as np

PRECISION_32 = 32
PRECISION_64 = 64

def matmul(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    return C, et - st

if __name__ == "__main__":
    N = 16384
    precision = PRECISION_32  # or PRECISION_64

    dtype = np.float32 if precision == PRECISION_32 else np.float64

    print(f"Using precision: {precision} bits ({dtype})")

    A = np.random.randn(N, N).astype(dtype)
    B = np.random.randn(N, N).astype(dtype)

    # Matrix multiplication involves two N x N matrices, resulting in:
    # N rows * N columns = N^2 elements

    # Each of these N^2 elements requires:
    # N multiplications (each element is the sum of N products)
    # N - 1 additions (summing N products, minus 1 since the first product is not an addition)

    # Since N is typically large, we can approximate:
    # N - 1 additions as N additions

    # Total number of operations:
    # N^2 elements * (N multiplications + N additions) = N^2 * 2N = 2 * N^3

    flop = 2 * N ** 3
    print(f"Total FLOP: {flop:.0f}")
    print(f"{flop / 1e9:.2f} GFLOP")

    _, s = matmul(A, B)
    print(f"Elapsed time for matmul: {s:.6f} seconds")

    gflops = (flop / s) / 1e9
    print(f"{gflops:.2f} GFLOPS")
