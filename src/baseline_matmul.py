import os
import sys
import time
import psutil

def generate_matrix(n, seed):
    matrix = [[0.0] * n for _ in range(n)]
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            row[j] = ((state >> 11) * (1.0 / 9007199254740992.0))
    return matrix

def identity_matrix(n):
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
    return matrix

def zero_matrix(n):
    return [[0.0] * n for _ in range(n)]

def multiply_kernel(A, B, C, n):
    for i in range(n):
        a_row = A[i]
        c_row = C[i]
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a_row[k] * B[k][j]
            c_row[j] = total

def compute_checksum(C, n):
    total = 0.0
    for i in range(n):
        row = C[i]
        for j in range(n):
            total += row[j]
    return total

def get_memory_kb():
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    if hasattr(mem_info, "peak_wset"):
        return mem_info.peak_wset / 1024.0
    return mem_info.rss / 1024.0

def main():
    if len(sys.argv) < 6:
        sys.exit(1)
    mode = sys.argv[1]
    n = int(sys.argv[2])
    warmup = int(sys.argv[3])
    reps = int(sys.argv[4])
    seed = int(sys.argv[5])

    if mode == "identity":
        A = generate_matrix(n, seed)
        B = identity_matrix(n)
        C = zero_matrix(n)
        multiply_kernel(A, B, C, n)
        max_diff = 0.0
        for i in range(n):
            for j in range(n):
                diff = abs(C[i][j] - A[i][j])
                if diff > max_diff:
                    max_diff = diff
        print(f"IDENTITY,{n},{max_diff:.17e},{compute_checksum(C, n):.17e}")
        return

    if mode == "zero":
        A = generate_matrix(n, seed)
        B = zero_matrix(n)
        C = zero_matrix(n)
        multiply_kernel(A, B, C, n)
        max_diff = 0.0
        for i in range(n):
            for j in range(n):
                diff = abs(C[i][j])
                if diff > max_diff:
                    max_diff = diff
        print(f"ZERO,{n},{max_diff:.17e},{compute_checksum(C, n):.17e}")
        return

    if mode == "dump":
        A = generate_matrix(n, seed)
        B = generate_matrix(n, seed + 1)
        C = zero_matrix(n)
        multiply_kernel(A, B, C, n)
        flat = [f"{C[i][j]:.17e}" for i in range(n) for j in range(n)]
        print(",".join(flat))
        return

    A = generate_matrix(n, seed)
    B = generate_matrix(n, seed + 1)
    C = zero_matrix(n)

    for _ in range(warmup):
        multiply_kernel(A, B, C, n)

    for r in range(1, reps + 1):
        for i in range(n):
            row = C[i]
            for j in range(n):
                row[j] = 0.0
        t0 = time.perf_counter_ns()
        multiply_kernel(A, B, C, n)
        t1 = time.perf_counter_ns()
        elapsed_s = (t1 - t0) / 1e9
        chk = compute_checksum(C, n)
        mem_kb = get_memory_kb()
        print(f"Python,{n},float64,{r},{elapsed_s:.9f},{mem_kb:.2f},{chk:.12e},1")

if __name__ == "__main__":
    main()
