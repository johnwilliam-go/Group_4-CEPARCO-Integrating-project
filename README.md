# Integrating project milestone #2
# Topic: Adaptive CUDA Histogram for Skewed Data

**Google Colab:** https://colab.research.google.com/drive/12aOsb-JMhCQ2kyEnV60-Jn_YYzwSKD0u?usp=sharing <br>
**Demo video:** https://drive.google.com/file/d/1J05qohf3dNnzeuW5Dy7QWJu-o9DDY7H3/view?usp=sharing

## Group Members

- OBAR, DAVID ANGELO
- GO, JOHN WILLIAM
- PLACER, PAUL JOHN
- ESPERANZA, MJ AJEGRAM

---

## Project Overview

A histogram is a data processing operation that determines the frequency of values within a defined range of bins. It is commonly used in applications such as image processing and data analysis.

This project analyzes how different data distributions affect the performance of GPU-based histogram operations. A sequential C implementation is compared with several CUDA implementations, including global-memory atomic operations and shared-memory histogram accumulation. An adaptive method that selects a counting strategy based on the estimated data distribution is also implemented and evaluated.

The implementation is tested on uniform, moderate-skew, and high-skew data across different input sizes and bin counts. Execution time, throughput, and speedup are measured, and the results include visualizations of how each implementation performs under different configurations.

## Inputs and Outputs

**Inputs**

- An array `x` of `N` integers, each in the range `[0, B)`.
- The number of bins `B`.
- The data distribution: uniform, moderate skew (Zipf s = 1.0), or high skew (Zipf s = 2.0).

**Outputs**

- An array `H` of `B` frequency counts, where `H[b]` = number of indices `i` in `[0, N-1]` with `x[i] = b`.
- For each run: execution time, throughput, speedup versus the sequential version, and a correctness check.

## Proposed Methods

The CPU handles input generation, memory allocation, distribution estimation, correctness checking, and timing. Five methods are compared:

1. **Sequential (C):** A single loop counts every element in the input array. It is the performance baseline and the correctness reference.
2. **Global Atomic (CUDA):** Each thread directly updates the global histogram using atomic operations. This is simple but slows down when multiple threads update the same bin.
3. **Shared Memory (CUDA):** Each block builds a local histogram in fast shared memory before merging it into the global histogram.
4. **Warp-Aggregated Shared Memory (CUDA):** Threads in a warp that update the same bin combine their counts first, so a single atomic update replaces up to 32 individual updates. This method was added beyond the original proposal.
5. **Adaptive (CUDA + CPU):** The CPU samples a small portion of the input to estimate the level of bin conflicts, then selects the kernel that best suits the estimated distribution (global atomic, shared memory, or warp-aggregated shared memory). The time spent on sampling and selection is included in the results.

The goal is to compare these methods across input sizes, bin counts, and data distributions, and to determine whether selecting a counting strategy based on the input distribution can improve performance over using a fixed method.

## Progress Update

### Completed

1. **Sequential C implementation:** the CPU baseline, also the reference for correctness checking.
2. **CUDA implementations**
   1. `global_atomic`: every thread does `atomicAdd` into the global histogram.
   2. `shared_memory`: per-block histogram in shared memory, merged into the global histogram at the end.
   3. `shared_warp_agg`: similar to `shared_memory`, but lanes of a warp that hit the same bin are combined first (`__match_any_sync`), so one atomic replaces up to 32 (needs compute capability 7.0+).
   4. `adaptive`: the CPU samples 8,192 elements to estimate the bin collision probability (p̂), then selects between global atomic, shared memory, and warp-aggregated shared memory.
      - It selects **global atomic** when `B > 12,288` or when `blocks × B > 5N`.
      - Otherwise it selects **warp-aggregated shared memory** when `p̂ ≥ 0.15`, and **plain shared memory** when `p̂ < 0.15`.
      - The selection time is fully included in kernel time and overlaps with the host-to-device transfer in total execution time.
3. **Correctness check:** every GPU result is compared bin by bin against the sequential C result, and the bin counts must sum to `N`. The check ran on every repetition of every applicable configuration (the shared-memory kernels do not run at 32,768 bins), and all checks passed.
4. **Performance measurements:** all methods were benchmarked on a Tesla T4 (sm_75) using input sizes from 2^18 to 2^26, bin counts of 256, 4096, 8192, and 32768, and uniform, moderate, and high-skew datasets. For each configuration, kernel time, total execution time (including host-to-device and device-to-host transfers), throughput, and speedup compared to the sequential C baseline were recorded.

### Results

Speedups are relative to the sequential C baseline. Throughput is in Gelem/s. At 32,768 bins the shared-memory kernels cannot run.

| Bins | Distribution | Method | Kernel speedup | Total speedup | Gelem/s |
|-----:|:-------------|:-------|---------------:|--------------:|--------:|
| 256 | high | adaptive | 40.45 | 3.16 | 34.34 |
| 256 | high | global_atomic | 2.05 | 1.28 | 1.79 |
| 256 | high | shared_memory | 43.66 | 3.14 | 37.28 |
| 256 | high | shared_warp_agg | 50.04 | 3.16 | 43.00 |
| 256 | moderate | adaptive | 25.71 | 1.97 | 33.88 |
| 256 | moderate | global_atomic | 1.24 | 0.78 | 1.67 |
| 256 | moderate | shared_memory | 28.54 | 1.96 | 37.83 |
| 256 | moderate | shared_warp_agg | 14.93 | 1.82 | 19.72 |
| 256 | uniform | adaptive | 41.34 | 3.49 | 32.00 |
| 256 | uniform | global_atomic | 2.89 | 1.60 | 2.25 |
| 256 | uniform | shared_memory | 54.21 | 3.49 | 42.02 |
| 256 | uniform | shared_warp_agg | 23.54 | 3.19 | 18.31 |
| 4096 | high | adaptive | 37.94 | 3.08 | 32.90 |
| 4096 | high | global_atomic | 2.06 | 1.27 | 1.80 |
| 4096 | high | shared_memory | 40.85 | 3.06 | 35.53 |
| 4096 | high | shared_warp_agg | 46.50 | 3.11 | 40.55 |
| 4096 | moderate | adaptive | 24.35 | 1.97 | 31.29 |
| 4096 | moderate | global_atomic | 1.69 | 0.95 | 2.20 |
| 4096 | moderate | shared_memory | 25.87 | 1.97 | 33.37 |
| 4096 | moderate | shared_warp_agg | 13.06 | 1.80 | 16.76 |
| 4096 | uniform | adaptive | 42.32 | 3.58 | 31.84 |
| 4096 | uniform | global_atomic | 27.15 | 3.38 | 20.58 |
| 4096 | uniform | shared_memory | 56.01 | 3.58 | 42.24 |
| 4096 | uniform | shared_warp_agg | 27.29 | 3.39 | 20.66 |
| 8192 | high | adaptive | 23.38 | 2.87 | 18.54 |
| 8192 | high | global_atomic | 2.01 | 1.28 | 1.64 |
| 8192 | high | shared_memory | 31.71 | 3.15 | 25.31 |
| 8192 | high | shared_warp_agg | 27.35 | 3.10 | 21.86 |
| 8192 | moderate | adaptive | 17.80 | 1.79 | 22.34 |
| 8192 | moderate | global_atomic | 1.78 | 0.97 | 2.35 |
| 8192 | moderate | shared_memory | 19.16 | 1.85 | 24.16 |
| 8192 | moderate | shared_warp_agg | 11.06 | 1.72 | 13.99 |
| 8192 | uniform | adaptive | 31.55 | 3.24 | 26.50 |
| 8192 | uniform | global_atomic | 26.26 | 3.15 | 21.80 |
| 8192 | uniform | shared_memory | 36.39 | 3.18 | 30.61 |
| 8192 | uniform | shared_warp_agg | 21.12 | 3.04 | 17.65 |
| 32768 | high | adaptive | 1.67 | 1.12 | 1.34 |
| 32768 | high | global_atomic | 1.50 | 1.04 | 1.19 |
| 32768 | moderate | adaptive | 2.12 | 1.08 | 2.66 |
| 32768 | moderate | global_atomic | 1.95 | 1.03 | 2.48 |
| 32768 | uniform | adaptive | 24.53 | 2.60 | 24.25 |
| 32768 | uniform | global_atomic | 24.72 | 2.61 | 24.46 |

## Initial Findings

### 1. Global atomic performs poorly under contention

- On moderate and high-skew data, global atomic achieves only around 1–2× kernel speedup over the CPU. It is also the only method that can be slower than the CPU overall (0.78× and 0.95× on moderate skew at 256 and 4096 bins).
- It performs best on uniform data with a higher number of bins, reaching around 25–27× speedup at 4096 bins and above.

### 2. Shared memory had the best overall performance

- It is fastest on uniform and moderate data (up to about 56× average kernel speedup).

### 3. Warp aggregation performs best only under high skew

- Its speedup is on par with or above shared memory at high skew (50.0× vs. 43.7× at 256 bins, 46.5× vs. 40.9× at 4096 bins).
- On uniform and moderate skew it reaches a lower speedup, roughly half that of shared memory (for example 14.9× vs. 28.5× on moderate data at 256 bins).
- At 8192 bins, plain shared memory is faster than warp aggregation even under high skew (31.7× vs. 27.4×).

### 4. Adaptive matches the best fixed method end-to-end but trails on kernel time

- Adaptive's total speedup is within about 1% of the best fixed method at 256 and 4096 bins (for example 3.16× vs. 3.16× on high skew at 256 bins). This is because the selection time overlaps with the host-to-device transfer. At 8192 bins it falls around 3–9% behind on skewed data (2.87× vs. 3.15× on high skew).
- In kernel time, adaptive is around 6–26% slower than the best fixed method, averaging roughly 16% across the nine configurations at 256–8192 bins, because the full selection time is included.

### 5. End-to-end speedup is much lower than kernel speedup

- Although kernel speedup reaches around 56×, total speedup is only around 0.8–3.6× because host-device transfers take up a significant amount of the execution time.

### 6. Large bin counts result in lower speedups

- At 32768 bins, only global atomics can run, resulting in around 1.5–2.1× speedup on skewed data.

## AI Usage

GenAI was used for idea generation such as literature review search, README formatting, and grammar checking.

## References

1. Milic, U., Gelado, I., Puzovic, N., Ramírez, A., & Tomasevic, M. (2013). Parallelizing general histogram application for CUDA architectures. In *Proceedings of the 2013 International Conference on Embedded Computer Systems: Architectures, Modeling, and Simulation (SAMOS XIII)* (pp. 11–18). IEEE. https://doi.org/10.1109/SAMOS.2013.6621100
2. Sakharnykh, N. (2015, March 17). GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell. *NVIDIA Technical Blog*. https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell
