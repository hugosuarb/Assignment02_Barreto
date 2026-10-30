# Assignment 02: Optimised Matrix Multiplication Approaches and Sparse Matrices

**Author:** Hugo Barreto Suárez\
**Course:** Big Data (3º Grado en Ciencia e Ingeniería de Datos -
ULPGC)\
**Academic Year:** 2026--2027\
**Report:**
[Assignment02_BarretoSuarez_Report.pdf](./Assignment02_BarretoSuarez_Report.pdf)

This repository contains the source code, benchmark scripts, raw
measurements, generated figures, and the final research report for
**Assignment 2**, which investigates dense matrix multiplication
optimisations (Part A: loop reordering and Strassen's algorithm) and
custom Compressed Sparse Row (CSR) sparse matrix multiplication across
varying sparsity levels (Part B) in Python.

## 1. Repository Structure

``` text
.
├── src/
│   ├── baseline_matmul.py                # Reused baseline i-j-k implementation from Assignment 1
│   └── matmul_a2.py                      # Implementations for Part A (ikj, Strassen) and Part B (Sparse CSR)
├── notebooks/
│   └── experiments_and_analysis.ipynb    # Automated build, validation, benchmark & plotting for Parts A & B
├── data/
│   ├── raw_measurements.csv              # Combined raw measurements for all configurations
│   ├── raw_part_a_dense.csv              # Raw measurements for dense optimisations
│   ├── raw_part_b_sparse.csv             # Raw measurements for sparse tests (CSR vs Dense)
│   ├── summary_part_a_dense.csv          # Aggregated medians, IQRs, and checksums for Part A
│   ├── summary_part_b_sparse.csv         # Aggregated medians, IQRs, and nonzeros (nnz) for Part B
│   └── execution_limits.csv              # Execution status (COMPLETED, TIMEOUT)
├── figures/
│   ├── part_a_dense_runtime.png          # Log-log plot of median kernel execution time (Part A)
│   ├── part_a_dense_memory.png           # Plot of median peak memory usage vs. matrix size (Part A)
│   └── part_b_sparse_vs_dense.png        # Plot of CSR vs dense execution times across sparsity levels (Part B)
├── Assignment02_BarretoSuarez_Report.pdf # Final research report (PDF)
└── README.md                             # Project documentation and instructions
```

*(Note: Make sure the source files are placed inside the `src/` folder
before running the notebook, or adjust the paths accordingly).*

## 2. Environment and Dependencies

The experiments reported in the PDF were executed on the following
setup:

-   **OS:** Windows 11 (64-bit)
-   **CPU:** Intel Core Ultra 7 155H (16 physical cores / 22 logical
    processors)
-   **RAM:** 15.37 GB
-   **Python:** CPython 3.14.6 (Anaconda environment)

### Python Packages Required

To run the Python script and the analysis notebook, install the required
libraries:

``` bash
pip install numpy pandas matplotlib psutil jupyter
```

## 3. Input Generation and LCG Determinism

To ensure fair comparisons and avoid disk I/O overhead, all inputs are
generated deterministically using a shared 64-bit Linear Congruential
Generator (LCG):

$$
\text{state} = (\text{state} \times 6364136223846793005 + 1442695040888963407) \pmod{2^{64}}
$$

-   **Part A (Dense):** Generates `n × n` dense matrices `A`
    (`seed = 12345`) and `B` (`seed = 12346`) with values in `[0, 1)`.
-   **Part B (Sparse):** Implements Bernoulli sampling via the same LCG:
    each matrix entry is non-zero with probability equal to the declared
    `density` parameter (storing values in `[0, 1)`).

## 4. Build and Run Instructions

The script `src/matmul_a2.py` accepts command-line arguments in the
following format:

``` text
<mode> <method> <n> <density> <warmup_runs> <measured_reps> <seed>
```

Where:

-   `<mode>` can be `dump` (prints flattened matrix `C` for NumPy
    correctness checks) or `benchmark` (runs warm-ups and measured
    repetitions, outputting CSV lines).
-   `<method>` can be `Dense_Baseline_ijk`, `Dense_LoopReorder_ikj`,
    `Dense_Strassen`, or `Sparse_CSR`.

### Run Individual Benchmarks Manually

Example for `Sparse_CSR` at `n = 400` with `5%` density (`95%`
sparsity), `1` warm-up run, `5` measured repetitions, and
`seed = 12345`:

``` bash
python src/matmul_a2.py benchmark Sparse_CSR 400 0.05 1 5 12345
```

## 5. Reproducing the Full Study and Plots

The easiest way to reproduce the entire experiment (correctness
validation at `n=20` against `numpy.matmul`, benchmarking for Part A
across `n ∈ {64, 128, 256, 384, 512}` and Part B across densities
`∈ {0.01, 0.05, 0.10, 0.25, 0.50}` at `n ∈ {200, 400}`, CSV aggregation,
and plot generation) is by running the Jupyter Notebook:

``` bash
jupyter notebook notebooks/experiments_and_analysis.ipynb
```

Execute all cells in order:

1.  **Cell 1:** Initializes environment variables and prints
    hardware/software details.
2.  **Cell 2:** Validates correctness across all implementations against
    NumPy (`rtol=1e-12`, `atol=1e-12`).
3.  **Cell 3:** Runs Part A dense optimisation benchmarks and saves raw
    CSV outputs.
4.  **Cell 4:** Runs Part B sparse matrix benchmarks (tracking
    conversion time and nonzeros `nnz`) and merges datasets.
5.  **Cell 5:** Computes medians and IQRs, saves summary CSV tables, and
    generates all figures in `figures/`.
