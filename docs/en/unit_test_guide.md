# Operator Optimization Unit Test Guide

## Test Introduction

`tests/test-sdpa-f16q80-opt` is a correctness unit test based on synthetic (mock) data, used to verify two inference paths.

| Path | ggml Operator |
| ------ | ----------- |
| Fused attention (SDPA) | `GGML_OP_FUSED_CPP_SDPA_EXT` (`ggml_fused_cpp_sdpa_ext`) |
| Matrix multiplication (GEMM) | `ggml_mul_mat` (F32/F16/Q8_0 weights) |

The test does not rely on any model files or external inputs. It generates synthetic tensors within the process, executes end-to-end inference through the actual GGML CPU computation graph, and compares the results against a C++ mathematical reference implementation.

## Test Coverage

### Fused SDPA (6 Cases)

The test parameters include `B`, `H`, `L`, `S`, `D`, `DV`, `scale`, and `mask`, covering multi-batch and multi-head configurations, under both masked and unmasked attention scenarios. Note that `DV` is constrained to be a multiple of 8 in all cases due to operator requirements.

```text
B=1 H=2 L=8  S=8  D=32  DV=32  scale=0.125  mask=0
B=1 H=2 L=8  S=8  D=32  DV=32  scale=0.125  mask=1
B=2 H=4 L=6  S=10 D=64  DV=64  scale=0.088  mask=0
B=2 H=4 L=6  S=10 D=64  DV=64  scale=0.088  mask=1
B=1 H=1 L=16 S=16 D=32  DV=32  scale=0.125  mask=0
B=1 H=1 L=16 S=16 D=32  DV=32  scale=0.125  mask=1
```

### GEMM (14 Cases)

**GEMM shapes are taken from the linear layers of real embedding models.**

- **bge-small-zh-v1.5**: `hidden=512, intermediate=2048`; attn Q/K/V/O: `M=512,K=512`; ffn_up: `M=2048,K=512`; ffn_down: `M=512,K=2048`.
- **bge-m3**: `hidden=1024, intermediate=4096`; attn: `M=1024,K=1024`; ffn_up: `M=4096,K=1024`; ffn_down: `M=1024,K=4096`.

```text
Weight  M    N    K    Layer (Model)
f32     512    8  512  attn(bge-small)
f16     512    8  512  attn(bge-small) -> FP16 NEON GEMM
q8_0    512    8  512  attn(bge-small) -> Q8_0 NEON GEMM
f16    2048    8  512  ffn_up(bge-small) -> FP16 NEON GEMM
q8_0   2048    8  512  ffn_up(bge-small) -> Q8_0 NEON GEMM
f16     512    8 2048  ffn_down(bge-small) -> FP16 NEON GEMM
q8_0    512    8 2048  ffn_down(bge-small) -> Q8_0 NEON GEMM
f32    1024    8 1024  attn(bge-m3)
f16    1024    8 1024  attn(bge-m3) -> FP16 NEON GEMM
q8_0   1024    8 1024  attn(bge-m3) -> Q8_0 NEON GEMM
f16    4096    8 1024  ffn_up(bge-m3) -> FP16 NEON GEMM
q8_0   4096    8 1024  ffn_up(bge-m3) -> Q8_0 NEON GEMM
f16    1024    8 4096  ffn_down(bge-m3) -> FP16 NEON GEMM
q8_0   1024    8 4096  ffn_down(bge-m3) -> Q8_0 NEON GEMM
```

### Key-Path Functions Involved in the Test

The test triggers the following optimized kernels (Arm/NEON) through the real ggml `mul_mat` and fused SDPA dispatches, ensuring that the **FP16 GEMM NEON** and **Q8_0 GEMM NEON operators** are actually exercised and verified.

| Path | Involved Function |
| ------ | ---------- |
| F16 GEMM (NEON) | `matmul_outer_packA_b_g_b16` -&gt; `matmul_outer_8x16_b_micro_packA_f16` / `matmul_outer_4x16_b_micro_packA_f16`; leftovers fall back to `ggml_matmul_f16_4x4_kernel` |
| Q8_0 GEMM (NEON) | `matmul_q8_0_mmla_spack_b_g` -&gt; `matmul_q8_0_mmla_8x8_b_micro_spack_neon` / `matmul_q8_0_mmla_4x4_b_micro_spack_neon` |
| F32 GEMM | `ggml_matmul_f32_4x4_kernel` |
| Fused SDPA | `ggml_compute_forward_fused_cpp_sdpa_ext` -&gt; `fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp` / `..._mask_f16` / `..._mask_f32` |

> ![icon-note](./public_sys-resources/icon-note.gif)**NOTE:** The `f16` and `q8_0` cases hit the NEON-accelerated GEMM kernels listed above; `f32` serves as the reference baseline for comparison.

## Test Principles

The test uses a seeded pseudo-random function to generate reproducible synthetic data, writes the data into ggml tensors, builds a computation graph, and executes it on the CPU backend. The expected results are then computed by a C++ reference implementation. Correctness is determined using the normalized mean square error (NMSE), defined as `nmse(a, b) = Σ(a - b)² / Σa²`. The acceptance thresholds are as follows: for fused SDPA, `NMSE < 1e-4`; for GEMM, thresholds are weight-type dependent—`F32` and `Q8_0` use `NMSE < 1e-4`, while `F16` uses `NMSE < 5e-3` (the FP16 accumulation precision decreases as K increases, so a looser threshold is applied for F16).

### SDPA Reference

1. `reference_sdpa()` reproduces `out = softmax(scale · Q·Kᵀ + mask) · V`.
2. Q/K/V/out tensors are accessed based on the contiguous memory index of the ggml tensor `ne` (innermost = `ne[0]`).
3. The mask is applied in the `[L, S]` layout (`mask[l*S + s]`).

### GEMM Reference

1. `reference_matmul()` reproduces the `ggml_mul_mat` convention.

   ```bash
   out[i][j] = Σ_k w[i][k] · act[j][k]
   ```

2. Activations are read in N×K layout (the right operand is treated as transposed); the output uses the raw layout `ne=[M, N]` with index `i + j·M`.

## Compilation

The fused SDPA operator is compiled into ggml-cpu only in Arm builds (`GGML_USE_FUSED_CPP_SDPA`). Therefore, the test participates in the build gated by architecture (`aarch64|arm`) in `tests/CMakeLists.txt`, and is compiled together with other tests. No additional CMake variables or model paths are required.

```bash
cmake -B build-delivery -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON -DBUILD_TESTING=ON

cmake --build build-delivery --target test-sdpa-f16q80-opt -j
cmake --build build-delivery -j
```

> ![icon-note](./public_sys-resources/icon-note.gif)**NOTE:** After adding a new test to an existing build directory, re-run `cmake .` to include it in the build system.

## Testing

```bash
build-delivery/bin/test-sdpa-f16q80-opt

ctest --test-dir build-delivery -L fused-sdpa
ctest --test-dir build-delivery -R test-sdpa-f16q80-opt
```

> ![icon-note](./public_sys-resources/icon-note.gif)**NOTE:** `0` is returned if all test cases pass. `1` is returned if any test fails.

## Expected Test Output

- `build-delivery/bin/test-sdpa-f16q80-opt`.

``` text
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

- `ctest --test-dir build-delivery -L fused-sdpa`.

```text
    Start 29: test-sdpa-f16q80-opt
1/1 Test #29: test-sdpa-f16q80-opt .............   Passed    0.59 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
fused-sdpa    =   0.59 sec*proc (1 test)

Total Test time (real) =   0.60 sec
```

- `ctest --test-dir build-delivery -R test-sdpa-f16q80-opt`.

```text
    Start 29: test-sdpa-f16q80-opt
1/1 Test #29: test-sdpa-f16q80-opt .............   Passed    0.62 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
fused-sdpa    =   0.62 sec*proc (1 test)

Total Test time (real) =   0.62 sec
```

## Change History

| Version | Date | Description |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | This is the first official release. |
