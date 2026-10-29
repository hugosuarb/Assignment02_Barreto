import os
import sys
import time
import psutil

def next_u64(state):
    return (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

def u64_to_double(state):
    return (state >> 11) * (1.0 / 9007199254740992.0)

def generate_dense_matrix(n, seed):
    matrix = [[0.0] * n for _ in range(n)]
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            state = next_u64(state)
            row[j] = u64_to_double(state)
    return matrix

def generate_sparse_logical_matrix(n, density, seed):
    matrix = [[0.0] * n for _ in range(n)]
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            state = next_u64(state)
            prob = u64_to_double(state)
            if prob < density:
                state = next_u64(state)
                val = u64_to_double(state)
                if val == 0.0:
                    val = 1.0
                row[j] = val
    return matrix

def zero_matrix(n):
    return [[0.0] * n for _ in range(n)]

def clear_matrix(C, n):
    for i in range(n):
        row = C[i]
        for j in range(n):
            row[j] = 0.0

def count_nnz_dense(M, n):
    count = 0
    for i in range(n):
        row = M[i]
        for j in range(n):
            if row[j] != 0.0:
                count += 1
    return count

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

def multiply_baseline_ijk(A, B, C, n):
    for i in range(n):
        a_row = A[i]
        c_row = C[i]
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a_row[k] * B[k][j]
            c_row[j] = total

def multiply_reordered_ikj(A, B, C, n):
    for i in range(n):
        a_row = A[i]
        c_row = C[i]
        for k in range(n):
            a_ik = a_row[k]
            if a_ik != 0.0:
                b_row = B[k]
                for j in range(n):
                    c_row[j] += a_ik * b_row[j]

def add_mat(A, B, m):
    R = [[0.0] * m for _ in range(m)]
    for i in range(m):
        r_row = R[i]
        a_row = A[i]
        b_row = B[i]
        for j in range(m):
            r_row[j] = a_row[j] + b_row[j]
    return R

def sub_mat(A, B, m):
    R = [[0.0] * m for _ in range(m)]
    for i in range(m):
        r_row = R[i]
        a_row = A[i]
        b_row = B[i]
        for j in range(m):
            r_row[j] = a_row[j] - b_row[j]
    return R

def strassen_recursive(A, B, m, leaf_size):
    if m <= leaf_size:
        C = zero_matrix(m)
        multiply_reordered_ikj(A, B, C, m)
        return C
    half = m // 2
    A11 = [row[:half] for row in A[:half]]
    A12 = [row[half:] for row in A[:half]]
    A21 = [row[:half] for row in A[half:]]
    A22 = [row[half:] for row in A[half:]]
    B11 = [row[:half] for row in B[:half]]
    B12 = [row[half:] for row in B[:half]]
    B21 = [row[:half] for row in B[half:]]
    B22 = [row[half:] for row in B[half:]]

    M1 = strassen_recursive(add_mat(A11, A22, half), add_mat(B11, B22, half), half, leaf_size)
    M2 = strassen_recursive(add_mat(A21, A22, half), B11, half, leaf_size)
    M3 = strassen_recursive(A11, sub_mat(B12, B22, half), half, leaf_size)
    M4 = strassen_recursive(A22, sub_mat(B21, B11, half), half, leaf_size)
    M5 = strassen_recursive(add_mat(A11, A12, half), B22, half, leaf_size)
    M6 = strassen_recursive(sub_mat(A21, A11, half), add_mat(B11, B12, half), half, leaf_size)
    M7 = strassen_recursive(sub_mat(A12, A22, half), add_mat(B21, B22, half), half, leaf_size)

    C11 = add_mat(sub_mat(add_mat(M1, M4, half), M5, half), M7, half)
    C12 = add_mat(M3, M5, half)
    C21 = add_mat(M2, M4, half)
    C22 = add_mat(add_mat(sub_mat(M1, M2, half), M3, half), M6, half)

    C = [[0.0] * m for _ in range(m)]
    for i in range(half):
        C[i][:half] = C11[i]
        C[i][half:] = C12[i]
    for i in range(half):
        C[i + half][:half] = C21[i]
        C[i + half][half:] = C22[i]
    return C

def next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p

def multiply_strassen(A, B, C, n, leaf_size=64):
    m = next_power_of_two(n)
    if m == n:
        R = strassen_recursive(A, B, n, leaf_size)
        for i in range(n):
            C[i][:] = R[i]
        return
    A_pad = zero_matrix(m)
    B_pad = zero_matrix(m)
    for i in range(n):
        A_pad[i][:n] = A[i]
        B_pad[i][:n] = B[i]
    R_pad = strassen_recursive(A_pad, B_pad, m, leaf_size)
    for i in range(n):
        C[i][:] = R_pad[i][:n]

def dense_to_csr(M, n):
    values = []
    col_indices = []
    row_ptr = [0]
    for i in range(n):
        row = M[i]
        for j in range(n):
            val = row[j]
            if val != 0.0:
                values.append(val)
                col_indices.append(j)
        row_ptr.append(len(values))
    return values, col_indices, row_ptr

def multiply_sparse_csr(A_csr, B_csr, C, n):
    a_val, a_col, a_row_ptr = A_csr
    b_val, b_col, b_row_ptr = B_csr
    for i in range(n):
        c_row = C[i]
        start_a = a_row_ptr[i]
        end_a = a_row_ptr[i + 1]
        for idx_a in range(start_a, end_a):
            k = a_col[idx_a]
            a_ik = a_val[idx_a]
            start_b = b_row_ptr[k]
            end_b = b_row_ptr[k + 1]
            for idx_b in range(start_b, end_b):
                j = b_col[idx_b]
                c_row[j] += a_ik * b_val[idx_b]

def run_method(method, A, B, C, n, A_csr=None, B_csr=None):
    if method == "Dense_Baseline_ijk":
        multiply_baseline_ijk(A, B, C, n)
    elif method == "Dense_LoopReorder_ikj":
        multiply_reordered_ikj(A, B, C, n)
    elif method == "Dense_Strassen":
        multiply_strassen(A, B, C, n, leaf_size=64)
    elif method == "Sparse_CSR":
        multiply_sparse_csr(A_csr, B_csr, C, n)
    else:
        sys.exit(1)

def main():
    if len(sys.argv) < 7:
        sys.exit(1)
    mode = sys.argv[1]
    method = sys.argv[2]
    n = int(sys.argv[3])
    density = float(sys.argv[4])
    warmup = int(sys.argv[5])
    reps = int(sys.argv[6])
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 12345

    if density >= 1.0:
        A = generate_dense_matrix(n, seed)
        B = generate_dense_matrix(n, seed + 1)
    else:
        A = generate_sparse_logical_matrix(n, density, seed)
        B = generate_sparse_logical_matrix(n, density, seed + 1)

    C = zero_matrix(n)

    if mode == "dump":
        A_csr = dense_to_csr(A, n) if method == "Sparse_CSR" else None
        B_csr = dense_to_csr(B, n) if method == "Sparse_CSR" else None
        run_method(method, A, B, C, n, A_csr, B_csr)
        flat = [f"{C[i][j]:.17e}" for i in range(n) for j in range(n)]
        print(",".join(flat))
        return

    conv_time_s = 0.0
    A_csr = None
    B_csr = None
    if method == "Sparse_CSR":
        t_c0 = time.perf_counter_ns()
        A_csr = dense_to_csr(A, n)
        B_csr = dense_to_csr(B, n)
        t_c1 = time.perf_counter_ns()
        conv_time_s = (t_c1 - t_c0) / 1e9
        nnz_a = len(A_csr[0])
        nnz_b = len(B_csr[0])
    else:
        nnz_a = count_nnz_dense(A, n)
        nnz_b = count_nnz_dense(B, n)

    for _ in range(warmup):
        clear_matrix(C, n)
        run_method(method, A, B, C, n, A_csr, B_csr)

    sparsity = round(1.0 - density, 4)
    rep_type = "CSR" if method == "Sparse_CSR" else "Dense_2D_List"

    for r in range(1, reps + 1):
        clear_matrix(C, n)
        t0 = time.perf_counter_ns()
        run_method(method, A, B, C, n, A_csr, B_csr)
        t1 = time.perf_counter_ns()
        kernel_s = (t1 - t0) / 1e9
        total_s = kernel_s + conv_time_s
        chk = compute_checksum(C, n)
        nnz_c = count_nnz_dense(C, n)
        mem_kb = get_memory_kb()
        print(f"{method},{rep_type},{n},float64,{density:.4f},{sparsity:.4f},{r},{kernel_s:.9f},{conv_time_s:.9f},{total_s:.9f},{mem_kb:.2f},{nnz_a},{nnz_b},{nnz_c},{chk:.12e},1")

if __name__ == "__main__":
    main()