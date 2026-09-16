# Feature Introduction

## Feature Description

llama-CPP is a patch collection that optimizes the official llama.cpp repository (commit `3ac67535c86`) for **Kunpeng 950 processors/new Kunpeng 920 processor models**. It leverages the Arm NEON, SVE-256, and i8mm instruction sets to optimize FP16, FP32, and Q8_0 matrix multiplication and fused Scaled Dot-Product Attention (SDPA) computation in the CPU backend of the llama.cpp inference engine. Built upon the llama.cpp computation graph scheduling framework, llama-CPP replaces the underlying kernels with in-house high-performance kernels via the GGML CPU backend's `type_traits` callback registration mechanism. Without changing the upper-level APIs and the computation graph structure, this optimization ensures full compatibility with non-optimized code execution paths.

## Architecture

| Layer | Optimization Status | Key Component |
| --- | --- | --- |
| Application layer | Unchanged | llama-embedding/llama-bench/llama-server |
| Framework layer | Partially changed | GGML OP IR (adding <code>GGML_OP_FUSED_CPP_SDPA_EXT</code>): <code>type_traits</code> MatMul routing + fused SDPA op dispatch |
| Kernel layer | Completely new | FP16/FP32/Q8_0 MatMul + fused SDPA (NEON/SVE/i8mm) |
| Hardware layer | Unchanged | NEON, SVE-256 (256-bit), and i8mm; OpenMP parallelism; 64-byte-aligned memory |

The data flow is as follows.

1. The application layer enters the inference loop through `llama_eval()`. The framework layer builds the GGML computation graph and schedules it via `ggml_graph_compute()`.

2. The CPU backend selects an optimized kernel based on `vec_dot_type`.

3. Packing and tiling are performed for Q8_0 data, and each kernel uses SVE-256/NEON/i8mm instructions for SIMD-accelerated computation.

## FP16 Matrix Multiplication Optimization

For an embedding model with F16 weights, matrix multiplication mainly occurs during the prefill stage and is the most frequent computation pattern. Four levels of kernels are provided to cover different M and N configurations.

### vec_dot_f16 (NEON Software-Pipelined Dot Product, General Fallback)

1. Initialize eight `float16x8_t` accumulators (`acc0..acc7`).

2. Preload the first 64 elements into `ax0..ax7`/`ay0..ay7`.

3. Execute the main loop computation as follows.

   ```bash
    (i=64; i<np; i+=64):
     acc[k] += ax[k] * ay[k]        
     ax[k] = vld1q_f16(&x[i+k*8])   
     ay[k] = vld1q_f16(&y[i+k*8])
    ```

4. Process the last 64 elements in the tail loop.

5. Execute `vpaddq_f16` and `vcvt_f32_f16`, and sum the result.

### FP16 4×4 Tile Kernel (NEON Outer Product)

1. For the `N==1` fast path: Loop over M rows with one `vfmaq_f16` execution per row to avoid function-call overhead.

2. For the 4×4 tile main loop.

   - Load `A[j..j+3][k..k+8]`, `B[i..i+3][k..k+8]`.

   - Perform 16 vfmaq_f16 operations (`j0×i0..i3`, `j1×i0..i3`, ...).

   - Execute `vcvt_f32_f16` and `vaddvq_f32`, and write the result to matrix C.

3. Batch process the M tail loop (1–3 rows); fall back to `vec_dot_f16` for the N tail loop.

>![icon-note](./public_sys-resources/icon-note.gif)**NOTE:**
>The FP16 4×4 tile kernel (NEON outer product) is suitable for small-batch prefill scenarios (M < 16).

### FP16 Outer-packA 8×16 Kernel (Hand-written NEON Outer Product for Large-Batch Prefill)

Input: packA (pre-packed A) with shape `[M_packed, K]` and B with shape `[N, K]` (both FP16). Constraints: `M%16==0` and `N%8==0`.

1. Main loop (N = 8, M = 16): Initialize 16 NEON accumulators (`v12...v27`).

2. K main loop (step = 32, double-block).

   - Load 16×2 A vectors (from packA) and 8 B vectors.

   - Broadcast each B vector lane and multiply each by the 16 A vectors -&gt; 8 × 16 = 128 fmla (256 in total due to step = 32)

3. Process the K full-block tail loop with step = 16; then process the K scalar tail loop with step = 2.

4. Execute fcvtl (f16 -&gt; f32) and write the result back to matrix C.

5. The implementation result is located in `ggml/src/ggml-cpu/vec.cpp`/`vec.h`.

>![icon-note](./public_sys-resources/icon-note.gif)**NOTE:**
>By pre-packing A into 8×16 tiles (packA), this kernel maximizes register utilization and instruction-level parallelism. It is ideal for large-batch prefill scenarios (M >= 16).

## FP32 Matrix Multiplication Optimization

For FP32 matrix multiplication, an SVE 4×4 tile kernel that leverages the 8-lane FP32 parallelism of 256-bit SVE is implemented to compute 16 (4×4) dot products at once.

1. N/M 4-aligned main loop: Initialize 16 `svfloat32_t` accumulators (`acc[0..3][0..3]`).

2. Tile K by the vector length (vl).

   - Load `A[j+0..3][k..k+vl]` into `j0..j3`.

   - Load `B[i+0..3][k..k+vl]` into `i0..i3`.

   - Perform `acc[m][n] += j[m] * i[n]` (16 `svmla_f32_m` operations).

3. Execute `svaddv` and write the result to matrix C; fall back to `ggml_vec_dot_f32` for the tail loop.

4. The implementation result is located in `ggml/src/ggml-cpu/vec.cpp`/`vec.h`.

## Q8_0 Quantized Matrix Multiplication Optimization (MMLA + `spack`)

The Q8_0 optimization delivers the most significant performance gains of this work. The core idea is to repack (`spack`) the `block_q8_0` data of matrices A and B according to the memory layout required by the MMLA (i8mm `smmla`, 8-bit × 8-bit -&gt; 32-bit) instruction, use the micro-kernel with SMMLA to compute the int8 dot product to obtain an int32 sum, multiply it by the A and B scales, and then accumulate the result into FP32.

Each Q8_0 block contains 32 int8 quantized values and one FP16 scale. The dot product of two blocks is computed as follows.

``` bash
Σk (da×qa[k]) × (db×qb[k]) = (da×db) × Σk qa[k]×qb[k]

int32_sum = Σ qa × qb            # SMMLA computation.
float_sum = int32_sum × scale_a × scale_b
C        += float_sum
```

### Packing Layout (`spack`)

- A is packed in groups of 2 rows: one Q8_0 block of every two rows of A is interleaved into four segments at +0/+16/+32/+48. In each 16-byte segment, the first 8 bytes are the 8 int8 values of row 0, and the last 8 bytes are the 8 int8 values of row 1. This exactly forms the 2-row × 8-K input required by SMMLA.

- B is packed in groups of 4 rows: one block of every four rows of B is packed into eight segments at +0/+16/+32/+48/+64/+80/+96/+112, corresponding to the combinations of cols0/1 and cols2/3 with `K0..7`, `K8..15`, `K16..23`, and `K24..31`.

- The same A panel is reused across multiple N tiles, avoiding repeated repacking of A.

### Tiling Strategy

Assume: outer tile: M=128 × N=64; K tile = 2048; micro tile = 8×8.

The complete execution flow of Q8_0 MatMul is as follows.

1. Zero out C.

2. Tile by M: Process at most 128 rows of A per iteration

3. Tile by K: Process at most 2048 K elements per iteration.

4. Pack the A panel (in groups of 2 rows).

5. Tile by N: Process at most 6 rows of B per iteration.

6. Pack the B panel (in groups of 4 rows).

7. Execute the micro-kernel (`matmul_q8_0_mmla_8x8_b_micro_spack_neon`).

   - Process one 8×8 C tile at a time (split into upper/lower 4×8 halves).

   - For each Q8_0 block, sequentially execute SMMLA on `K0..7`, `K8..15`, `K16..23`, and `K24..31`. Then convert the resulting int32 sum to FP32, multiply it by both the A scale and the B scale, and then accumulate the final FP32 product into C.

The call flow of Q8_0 MatMul is as follows.

1. `ggml_compute_forward_mul_mat()` calls `compute_forward_mul_mat_one_chunk_matmul_kernel()`.

2. This function routes to the `matmul_q8_0_mmla_spack_b_g()` callback via `type_traits[Q8_0].matmul`.

3. `matmul_q8_0_mmla_spack_b()` performs outer M/N/K tiling and A/B panel packing.

4. Finally, `matmul_q8_0_mmla_8x8_b_micro_spack_neon()` is called to complete the micro-kernel computation.

>![icon-note](./public_sys-resources/icon-note.gif)**NOTE:**
>`type_traits[GGML_TYPE_Q8_0]` registers the `matmul_q8_0_mmla_spack_b_g` callback, and falls back to the generic kernel when the rows/columns do not satisfy tile alignment. The implementation is located in `ggml/src/ggml-cpu/ggml-cpu-quants.c` and `ggml/src/ggml-cpu/ggml-cpu-quants.h`, with routing registered in `ggml-cpu.c`.

## Fused SDPA (FlashAttention v2 NEON Fused Operator)

The official llama.cpp attention implementation executes QK<sup>T</sup>, scaling, masking, Softmax, and PV as separate operators. The intermediate scores matrix is repeatedly read from and written to memory, which on CPUs often shifts the bottleneck from compute power to L2/L3/memory bandwidth. Fused SDPA combines these steps into a single C++/NEON/SVE kernel to process data in tiles in a streaming fashion, thereby reducing intermediate tensors and memory traffic.

### Integration Method

This feature is controlled by the `GGML_USE_FUSED_CPP_SDPA` compilation macro. `build_attn_mha()` in `llama-graph.cpp` checks the following enabling conditions.

- AArch64 platform; F32 Q/K/V; MHA (`n_head==n_head_kv`).

- No MLA/ALiBi/attn_soft_cap.

- Contiguous or absent mask (F16/F32); `v->ne[0] % 8 == 0`.

When the conditions are met, `GGML_OP_FUSED_CPP_SDPA_EXT` is built to replace the official flash_attn.

### Inference Flow

The inference flow is as follows.

1. `build_attn_mha()` is used to determine whether to use the fused SDPA path.

2. When the enabling conditions are met, `GGML_OP_FUSED_CPP_SDPA_EXT` is built and `ggml_fused_cpp_sdpa_ext(q, k, v, mask, scale)` is called, which performs the following operations through `ggml_compute_forward_fused_cpp_sdpa_ext()` in `ops.cpp`.

   - If `L == S` and `L % 64 == 0`, it adopts diagonal tiling (64×64 blocks), computing only the causal upper triangle.

   - Otherwise, `fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp()` is called for full-matrix computation.

3. When the conditions are not met, it falls back to the official flash_attn (`flash_attn_ext`).

### Diagonal Tiling Optimization

When `L == S` (prefill) and `L % 64 == 0`, the attention matrix is partitioned into 64×64 blocks along the diagonal. Only the blocks lying in the causal upper triangle are computed, reducing the overall computational workload.

### NEON Kernel

- Completes all Q/K/V processing within the micro-kernel.

- Performs packing and transposition in the micro-kernel to avoid additional memory movement.

- Adopts an L3-cache-aware KV tiling strategy.

- Provides three variants (mask_f16/mask_f32/no mask) with head-level thread partitioning for parallelism.

- The implementation is located in `ggml/src/ggml-cpu/fused-cpp/fp32_packqkv/`, with the entry point in `ops.cpp`, and the operator definition in `ggml.h` and `ggml.c`.

## Key Interfaces

```c
void ggml_vec_dot_f16(int n, float * s, size_t bs,
                      ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
void ggml_matmul_f16_4x4_kernel(size_t na, size_t nb, int dim,
                                const void * a, const void * b, void * c, size_t c_column_bytes);
void ggml_matmul_f32_4x4_kernel(size_t na, size_t nb, int dim,
                                const void * a, const void * b, void * c, size_t c_column_bytes);
void matmul_outer_8x16_b_micro_packA_f16(size_t M, size_t N, size_t K,
    const void * a, size_t a_column_bytes, const void * b, size_t b_column_bytes,
    void * c, size_t c_column_bytes);

void matmul_q8_0_mmla_spack_padded(bool isA, size_t nr_real, size_t nr_pack, int dim,
    const block_q8_0 * X, size_t column_bytes, matmul_q8_0_mmla_spack_t ** out);
void matmul_q8_0_mmla_4x4_tiled_padded(size_t na_real, size_t nb_real,
    size_t na_pad, size_t nb_pad, int K, size_t offM, size_t offN, size_t offK,
    const block_q8_0 * A, const block_q8_0 * B, size_t a_stride, size_t b_stride,
    size_t a_col_bytes, size_t b_col_bytes, float * tmp_c,
    matmul_q8_0_mmla_spack_t ** pack_a, matmul_q8_0_mmla_spack_t ** pack_b);

struct ggml_tensor * ggml_fused_cpp_sdpa_ext(struct ggml_context * ctx,
    struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v,
    struct ggml_tensor * mask, float scale);
void ggml_compute_forward_fused_cpp_sdpa_ext(const struct ggml_compute_params * params,
    struct ggml_tensor * dst);
```

## Fixes for llama.cpp Issues

The optimization patches also include several functional fixes to the official llama.cpp source code to ensure fairness in accuracy verification.

- `common_embd_normalize`/`common_embd_similarity_cos` are changed to double-precision accumulation to avoid FP32 overflow on long sequences.

- `common` links OpenMP when `GGML_OPENMP` is enabled (required by `omp simd`).

- `tools/server` fixes the trailing whitespace issue in the embedding output path, and adds an embedding buffer using the `reserve()` helper function.

## Compilation and Running

### Compilation Options

- Fused SDPA compilation (recommended, for enabling the fused SDPA path)

```bash
-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
CFLAGS+=-DGGML_USE_FUSED_CPP_SDPA
```

- Standard compilation (matrix multiplication optimization only, excluding fused SDPA)

```bash
-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
```

- One-click compilation

   ```bash
   ./compile.sh build-delivery
   ```

### Environment Variable Control

| Environment Variable | Value | Effect |
| --- | --- | --- |
| GGML_FUSED_CPP_SDPA | <code>0</code>/<code>off</code>/<code>false</code>/<code>no</code> | Disables fused SDPA and falls back to the official flash_attn. |
| GGML_FUSED_CPP_SDPA | <code>debug</code>/<code>trace</code> | Enables fused SDPA debug logging. |
| GGML_FUSED_CPP_SDPA | Empty/Other | Enabled by default. |

## Acceptance Criteria

- Functional correctness: After optimization, the cosine similarity between each kernel's output and the official llama.cpp output on the specified test dataset is greater than 0.999, or the average score drop across all C-MTEB datasets is less than 1%, and the result passes the script-based accuracy verification.

- Performance: After optimization, each kernel achieves the expected end-to-end speedup in benchmark tests (for platform-specific performance data, see [Technical Report](./technical_report.md)).

## Change History

| Version | Date | Description |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | This is the first official release. |
