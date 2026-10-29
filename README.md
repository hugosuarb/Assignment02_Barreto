# Assignment 02: Optimised Matrix Multiplication Approaches and Sparse Matrices

Empirical investigation of dense matrix multiplication optimisations (Part A) and Compressed Sparse Row (CSR) sparse matrix multiplication across varying sparsity levels (Part B) in Python.

## Reused Materials from Assignment 01

* `src/baseline_matmul.py`: Reused validated basic $O(n^3)$ triple-loop (`i-j-k`) Python implementation and 64-bit LCG input generator from Assignment 01, serving as the sequential baseline (`Dense_Baseline_ijk`).

## Repository Structure

* `src/matmul_a2.py`: Implementations for Part A (`Dense_Baseline_ijk`, `Dense_LoopReorder_ikj`, `Dense_Strassen`) and Part B (`Sparse_CSR`), including conversion timing and non-zero (`nnz`) tracking.
* `notebooks/experiments_and_analysis.ipynb`: Automated correctness verification against NumPy, benchmarking harness for Part A and Part B, statistical aggregation (median and IQR), and plot generation.
* `data/raw_measurements.csv`: Combined raw measurements for all configurations and repetitions.
* `data/summary_part_a_dense.csv` & `data/summary_part_b_sparse.csv`: Aggregated tables for the report.
* `figures/`: Generated plots (`part_a_dense_runtime.png`, `part_a_dense_memory.png`, `part_b_sparse_vs_dense.png`).

## Environment and Dependencies

* Python 3.x with `numpy`, `pandas`, `matplotlib`, `psutil`, and `jupyter`.

## Exact Run Commands

### Run Individual Benchmarks Manually

Arguments: `<mode> <method> <n> <density> <warmup_runs> <measured_reps> <seed>`

```bash
python src/matmul_a2.py benchmark Dense_Baseline_ijk 256 1.0 1 5 12345
python src/matmul_a2.py benchmark Dense_LoopReorder_ikj 256 1.0 1 5 12345
python src/matmul_a2.py benchmark Dense_Strassen 256 1.0 1 5 12345
python src/matmul_a2.py benchmark Sparse_CSR 400 0.05 1 5 12345