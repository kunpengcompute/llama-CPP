# Operator Optimization Unit Test Guide

## 1. Overview

`tests/test-sdpa-f16q80-opt` is a correctness unit test based on synthetic (mock) data, used to verify two inference paths:

| Path | ggml Operator |
| --- | --- |
| Fused attention (SDPA) | `GGML_OP_FUSED_CPP_SDPA_EXT` (`ggml_fused_cpp_sdpa_ext`) |
| Matrix multiplication (GEMM) | `ggml_mul_mat` (F32 / F16 / Q8_0 weights) |

The test does not depend on model files or any external input: synthetic tensors are generated in-process, executed end-to-end through the real ggml CPU compute graph, and compared against a C++ math reference implementation.

## 2. Coverage

### 2.1 fused SDPA (6 cases)

Parameters `(B, H, L, S, D, DV, scale, mask)`:

```
B=1 H=2 L=8  S=8  D=32  DV=32  scale=0.125  mask=0
B=1 H=2 L=8  S=8  D=32  DV=32  scale=0.125  mask=1
B=2 H=4 L=6  S=10 D=64  DV=64  scale=0.088  mask=0
B=2 H=4 L=6  S=10 D=64  DV=64  scale=0.088  mask=1
B=1 H=1 L=16 S=16 D=32  DV=32  scale=0.125  mask=0
B=1 H=1 L=16 S=16 D=32  DV=32  scale=0.125  mask=1
```

Covers multiple batches, multiple heads, and with/without mask; `DV` is always a multiple of 8 (operator constraint).

### 2.2 GEMM (14 cases)

**GEMM shapes are taken from the linear layers of real embedding models:**
- **bge-small-zh-v1.5**: `hidden=512, intermediate=2048` → attn Q/K/V/O `M=512,K=512`, ffn_up `M=2048,K=512`, ffn_down `M=512,K=2048`
- **bge-m3**: `hidden=1024, intermediate=4096` → attn `M=1024,K=1024`, ffn_up `M=4096,K=1024`, ffn_down `M=1024,K=4096`

```
Weight type  M    N    K    Corresponding layer (model)
f32        512    8  512  attn (bge-small)
f16        512    8  512  attn (bge-small) → FP16 NEON GEMM
q8_0       512    8  512  attn (bge-small) → Q8_0 NEON GEMM
f16       2048    8  512  ffn_up (bge-small) → FP16 NEON GEMM
q8_0      2048    8  512  ffn_up (bge-small) → Q8_0 NEON GEMM
f16        512    8 2048  ffn_down (bge-small) → FP16 NEON GEMM
q8_0       512    8 2048  ffn_down (bge-small) → Q8_0 NEON GEMM
f32       1024    8 1024  attn (bge-m3)
f16       1024    8 1024  attn (bge-m3) → FP16 NEON GEMM
q8_0      1024    8 1024  attn (bge-m3) → Q8_0 NEON GEMM
f16       4096    8 1024  ffn_up (bge-m3) → FP16 NEON GEMM
q8_0      4096    8 1024  ffn_up (bge-m3) → Q8_0 NEON GEMM
f16       1024    8 4096  ffn_down (bge-m3) → FP16 NEON GEMM
q8_0      1024    8 4096  ffn_down (bge-m3) → Q8_0 NEON GEMM
```

### 2.3 Key functions/paths covered

The test triggers the following optimized kernels (ARM/NEON) through the real ggml `mul_mat` / fused SDPA dispatch, ensuring that the **FP16 GEMM NEON operator** and the **Q8_0 GEMM NEON operator** are actually executed and verified:

| Path | Functions Involved |
| --- | --- |
| F16 GEMM (NEON) | `matmul_outer_packA_b_g_b16` → `matmul_outer_8x16_b_micro_packA_f16` / `matmul_outer_4x16_b_micro_packA_f16`; leftover goes through `ggml_matmul_f16_4x4_kernel` |
| Q8_0 GEMM (NEON) | `matmul_q8_0_mmla_spack_b_g` → `matmul_q8_0_mmla_8x8_b_micro_spack_neon` / `matmul_q8_0_mmla_4x4_b_micro_spack_neon` |
| F32 GEMM | `ggml_matmul_f32_4x4_kernel` |
| Fused SDPA | `ggml_compute_forward_fused_cpp_sdpa_ext` → `fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp` / `..._mask_f16` / `..._mask_f32` |

> **Key coverage**: the `f16` and `q8_0` cases hit the NEON GEMM accelerated kernels above; the `f32` cases serve as the reference baseline for comparison.

## 3. Principle

1. Generate synthetic data with a seeded pseudo-random function (reproducible).
2. Write the data into ggml tensors, build the compute graph, and execute on the CPU backend.
3. A C++ reference implementation computes the expected results.
4. Judge correctness with the normalized mean square error (NMSE):

   ```
   nmse(a, b) = Σ(a - b)² / Σa²
   ```

   Thresholds: fused SDPA `NMSE < 1e-4`; GEMM differs by weight type — F32/Q8_0 `NMSE < 1e-4`, F16 `NMSE < 5e-3` (FP16 accumulation precision degrades as K grows, so the F16 threshold is relaxed).

### 3.1 SDPA reference

`reference_sdpa()` reproduces `out = softmax(scale · Q·Kᵀ + mask) · V`. q/k/v/out are accessed by the contiguous memory index of the ggml tensor `ne` (innermost = `ne[0]`); the mask is applied in the `[L, S]` layout (`mask[l*S + s]`).

### 3.2 GEMM reference

`reference_matmul()` reproduces the `ggml_mul_mat` convention:

```
out[i][j] = Σ_k w[i][k] · act[j][k]
```

Activations are read in N×K layout (the right operand is treated as transposed); the output uses the raw layout `ne=[M, N]` with index `i + j·M`.

## 4. Build

The fused SDPA operator is only compiled into ggml-cpu on ARM builds (`GGML_USE_FUSED_CPP_SDPA`); the test therefore participates in the build gated by architecture (`aarch64|arm`) in `tests/CMakeLists.txt`, and is compiled together with the tests. No additional CMake variables or model paths are needed.

```bash
cmake -B build-delivery -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON -DBUILD_TESTING=ON

cmake --build build-delivery --target test-sdpa-f16q80-opt -j
# or
cmake --build build-delivery -j
```

> After adding a new test to an existing build directory, re-run `cmake .` so that it is included in the build system.

## 5. Run

```bash
# Run directly (no arguments required)
build-delivery/bin/test-sdpa-f16q80-opt

# Via ctest (LABEL = fused-sdpa)
ctest --test-dir build-delivery -L fused-sdpa
ctest --test-dir build-delivery -R test-sdpa-f16q80-opt
```

All cases passing returns 0; any failure returns 1.

## 6. Expected Output
- `build-delivery/bin/test-sdpa-f16q80-opt`
```
=== fused SDPA (GGML_OP_FUSED_CPP_SDPA_EXT) ===
  SDPA B=1 H=2 L=8 S=8 D=32 DV=32 mask=0 NMSE=6.010e-11 ok
  SDPA B=1 H=2 L=8 S=8 D=32 DV=32 mask=1 NMSE=7.238e-11 ok
  SDPA B=2 H=4 L=6 S=10 D=64 DV=64 mask=0 NMSE=7.822e-11 ok
  SDPA B=2 H=4 L=6 S=10 D=64 DV=64 mask=1 NMSE=1.039e-10 ok
  SDPA B=1 H=1 L=16 S=16 D=32 DV=32 mask=0 NMSE=2.866e-10 ok
  SDPA B=1 H=1 L=16 S=16 D=32 DV=32 mask=1 NMSE=3.228e-10 ok

=== GEMM (ggml_mul_mat, mock) ===
  GEMM f32   M=512 N=8 K=512  NMSE=3.476e-15 ok
  GEMM f16   M=512 N=8 K=512  NMSE=1.063e-05 ok
  GEMM q8_0  M=512 N=8 K=512  NMSE=4.186e-08 ok
  GEMM f16   M=2048 N=8 K=512 NMSE=1.056e-05 ok
  GEMM q8_0  M=2048 N=8 K=512 NMSE=4.203e-08 ok
  GEMM f16   M=512 N=8 K=2048 NMSE=2.575e-04 ok
  GEMM q8_0  M=512 N=8 K=2048 NMSE=3.769e-08 ok
  GEMM f32   M=1024 N=8 K=1024 NMSE=5.007e-15 ok
  GEMM f16   M=1024 N=8 K=1024 NMSE=5.758e-05 ok
  GEMM q8_0  M=1024 N=8 K=1024 NMSE=3.266e-08 ok
  GEMM f16   M=4096 N=8 K=1024 NMSE=5.705e-05 ok
  GEMM q8_0  M=4096 N=8 K=1024 NMSE=3.266e-08 ok
  GEMM f16   M=1024 N=8 K=4096 NMSE=1.790e-03 ok
  GEMM q8_0  M=1024 N=8 K=4096 NMSE=2.697e-08 ok

=== 0 cases failed ===
```
- `ctest --test-dir build-delivery -L fused-sdpa`
```
    Start 29: test-sdpa-f16q80-opt
1/1 Test #29: test-sdpa-f16q80-opt .............   Passed    0.59 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
fused-sdpa    =   0.59 sec*proc (1 test)

Total Test time (real) =   0.60 sec
```
- `ctest --test-dir build-delivery -R test-sdpa-f16q80-opt`
```
    Start 29: test-sdpa-f16q80-opt
1/1 Test #29: test-sdpa-f16q80-opt .............   Passed    0.62 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
fused-sdpa    =   0.62 sec*proc (1 test)

Total Test time (real) =   0.62 sec
```
