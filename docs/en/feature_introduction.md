# Feature Introduction

The llama-CPP baseline is based on the official llama.cpp commit `3ac67535c86`. This optimization targets **Kunpeng processors**, using ARM NEON / SVE-256 / i8mm instructions, and focuses on the matrix multiplication and attention (SDPA) operators in the llama.cpp inference engine CPU backend. It covers FP16, FP32, and Q8_0 matrix multiplication as well as fused attention computation. It does not modify the upper-layer APIs or the compute graph structure; instead, it replaces the underlying kernels through the `type_traits` callback registration mechanism of the GGML CPU backend, remaining fully compatible with the non-optimized path.

## Overall Architecture

The optimization builds on the compute-graph scheduling mechanism of llama.cpp; it injects purpose-built high-performance kernels into the GGML CPU backend, covering FP16, FP32, and Q8_0 matrix multiplication plus fused SDPA (attention).

| Layer | Optimization Status | Key Components |
| --- | --- | --- |
| Application | Unmodified | llama-embedding / llama-bench / llama-server |
| Framework | Partially modified | GGML OP IR (new GGML_OP_FUSED_CPP_SDPA_EXT) → type_traits matmul routing + fused SDPA op dispatch |
| Kernel (this work) | All new | FP16 / FP32 / Q8_0 MatMul + Fused SDPA (NEON / SVE / i8mm) |
| Hardware | Unmodified | NEON, SVE-256 (256-bit), i8mm; OpenMP parallelism; 64B-aligned memory |

Data flow: the application enters the inference loop through `llama_eval()` → the framework builds the GGML graph and schedules it with `ggml_graph_compute()` → the CPU backend selects the optimized kernel according to `vec_dot_type` (F32/F16 → matmul kernels in `ops.cpp` or the fused SDPA entry; Q8_0 → MMLA spack packing + tile computation) → each kernel uses SVE-256 / NEON / i8mm instructions for SIMD-accelerated computation.

## FP16 Matrix Multiplication Optimization

When model weights are F16, matrix multiplication mainly occurs in the prefill phase and is the most frequent computation pattern. Four levels of kernels are provided to cover different M and N configurations.

### vec_dot_f16 (NEON software-pipelined dot product, general fallback)

```bash
1. Initialize 8 float16x8_t accumulators (acc0..acc7)
2. Prefetch the first 64 elements into ax0..ax7 / ay0..ay7
3. Main loop (i=64; i<np; i+=64):
     acc[k] += ax[k] * ay[k]        # compute this round
     ax[k] = vld1q_f16(&x[i+k*8])   # prefetch the next round (software pipeline)
     ay[k] = vld1q_f16(&y[i+k*8])
4. Tail loop for the last 64 elements
5. vpaddq_f16 horizontal add + vcvt_f32_f16 to f32 sum
```

### FP16 4×4 tile kernel (NEON outer product)

```bash
1. N==1 fast path: loop over M rows with one vfmaq_f16 accumulate per row
2. 4×4 tile main loop:
   - load A[j..j+3][k..k+8], B[i..i+3][k..k+8]
   - 16 vfmaq_f16 outer-product accumulates (j0×i0..i3, j1×i0..i3, ...)
   - horizontal add vcvt_f32_f16 + vaddvq_f32 write to C
3. M-tail (1-3 rows) batch processed; N-tail falls back to vec_dot_f16
```

Suited to small prefill batches (M<16).

### FP16 outer-packA 8×16 kernel (hand-written NEON outer product, large prefill batches)

```bash
Input: packA (prepacked A)[M_packed, K], B[N,K] (fp16); constraints M%16==0, N%8==0
1. Main loop (N in 8s, M in 16s): initialize 16 NEON accumulators (v12..v27)
2. K main loop (step=32, double block):
   - load 16×2 A vectors (from packA), 8 B vectors
   - each B vector lane is broadcast-multiplied by 16 A vectors → 8 × 16 = 128 fmla (256 total for step=32)
3. K full-block tail loop (step=16); K scalar tail loop (step=2)
4. fcvtl f16→f32 write back to C
```

A is prepacked (packA) in 8×16 tiles, fully exploiting registers and instruction-level parallelism; suited to large-batch prefill (M>=16).

Implemented in `ggml/src/ggml-cpu/vec.cpp` / `vec.h`.

## FP32 Matrix Multiplication Optimization

An SVE 4×4 tile kernel is implemented using the 8-lane FP32 parallelism of 256-bit SVE, computing 4×4=16 dot products at once:

```bash
1. N/M 4-aligned main loop: initialize 16 svfloat32_t accumulators (acc[0..3][0..3])
2. Chunk K by vl (vector length):
   - load A[j+0..3][k..k+vl] into j0..j3
   - load B[i+0..3][k..k+vl] into i0..i3
   - acc[m][n] += j[m] * i[n] (16 svmla_f32_m)
3. svaddv horizontal sum into C; tail falls back to ggml_vec_dot_f32
```

Implemented in `ggml/src/ggml-cpu/vec.cpp` / `vec.h`.

## Q8_0 Quantized Matrix Multiplication Optimization (MMLA + Spack)

The Q8_0 optimization delivers the largest performance gain of this work. Core idea: repack (spack) the `block_q8_0` data of matrices A and B into the memory layout required by the MMLA (`smmla`, 8-bit×8-bit→32-bit) instruction, then use the micro-kernel with SMMLA to compute the int8 dot product into int32, then multiply by the A/B scales and accumulate into FP32.

Each Q8_0 block contains 32 int8 quantized values and one fp16 scale. The dot product of two blocks is:

```bash
Σk (da×qa[k]) × (db×qb[k]) = (da×db) × Σk qa[k]×qb[k]

int32_sum = Σ qa × qb            # computed with SMMLA
float_sum = int32_sum × scale_a × scale_b
C        += float_sum
```

### Packing layout (Spack)

- A is packed in groups of 2 rows: one Q8_0 block of every two A rows is packed into four segments +0/+16/+32/+48, each 16 bytes with the first 8 bytes being the 8 int8 values of row 0 and the last 8 bytes the 8 int8 values of row 1, exactly forming the 2-row × 8-K input needed by SMMLA;
- B is packed in groups of 4 rows: one block of every four B rows is packed into eight segments +0/+16/+32/+48/+64/+80/+96/+112, corresponding to the combinations of cols0/1, cols2/3 and K0..7, 8..15, 16..23, 24..31;
- The same A panel is reused across multiple N tiles, avoiding repeated packing of A.

### Tiling strategy

Outer tile M=128 × N=64, K tile=2048, micro tile 8×8. The full flow:

```bash
Full Q8_0 MatMul
├─ Zero C
├─ M tiling: up to 128 A rows each time
│  └─ K tiling: up to 2048 each time
│     ├─ pack A panel (groups of 2 rows)
│     └─ N tiling: up to 64 B rows each time
│        ├─ pack B panel (groups of 4 rows)
│        └─ micro-kernel (matmul_q8_0_mmla_8x8_b_micro_spack_neon)
│           ├─ processes one 8×8 C tile (split into upper/lower 4×8)
│           └─ each Q8_0 block: K0..7 / K8..15 / K16..23 / K24..31 → SMMLA
│              int32→fp32 → ×A scale → ×B scale → accumulate to C
```

Call chain: `ggml_compute_forward_mul_mat()` → `compute_forward_mul_mat_one_chunk_matmul_kernel()` → `type_traits[Q8_0].matmul = matmul_q8_0_mmla_spack_b_g()` → `matmul_q8_0_mmla_spack_b()` (outer M/N/K tiling, packing A/B panels) → `matmul_q8_0_mmla_8x8_b_micro_spack_neon()`.

`type_traits[GGML_TYPE_Q8_0]` registers the `matmul_q8_0_mmla_spack_b_g` callback, falling back to the generic kernel when rows/columns are not tile-aligned. Implemented in `ggml/src/ggml-cpu/ggml-cpu-quants.c/.h`, with routing registered in `ggml-cpu.c`.

## Fused SDPA (FlashAttention v2 NEON Fused Operator)

The native llama.cpp attention executes QKᵀ, scaling, mask, softmax, and PV as separate operators, repeatedly reading and writing the large intermediate scores matrix. On the CPU this easily shifts the workload from a compute bottleneck to an L2/L3/memory-bandwidth bottleneck. Fused SDPA merges these steps into a single dedicated C++/NEON/SVE kernel that streams by tile, reducing intermediate tensors and memory traffic.

**Integration**: controlled by the `GGML_USE_FUSED_CPP_SDPA` compile flag. `build_attn_mha()` in `llama-graph.cpp` detects the enabling conditions:

- ARM64 platform; q/k/v all F32; MHA (n_head == n_head_kv);
- no MLA, no ALiBi, no attn_soft_cap;
- mask is F16/F32 contiguous or absent; v->ne[0] % 8 == 0.

When satisfied, the `GGML_OP_FUSED_CPP_SDPA_EXT` op is built to replace the native flash_attn.

### Inference flow

```bash
build_attn_mha()
├─ use_fused_cpp_attn?
│  ├─ YES → ggml_fused_cpp_sdpa_ext(q, k, v, mask, scale)
│  │        → ops.cpp: ggml_compute_forward_fused_cpp_sdpa_ext()
│  │           partition heads across threads
│  │          ├─ L == S and L%64==0 → diagonal blocks (64×64, causal upper triangle only)
│  │          └─ else → full-matrix call
│  │              fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp()
│  └─ NO → native flash_attn (flash_attn_ext)
```

**Diagonal-block optimization**: when `L == S` (prefill) and `L % 64 == 0`, the attention matrix is divided along the diagonal into 64×64 blocks, computing only the upper-triangular blocks inside the causal mask to reduce work.

**NEON kernel**: Q/K/V are packed and transposed inside the micro-kernel to avoid extra memory movement; an L3-cache-aware KV tiling strategy is used; FP32 internal precision with pb16 approximate softmax; three variants are provided (mask_f16 / mask_f32 / maskless) with head-level thread partitioning for parallelism.

Implemented in `ggml/src/ggml-cpu/fused-cpp/fp32_packqkv/`, with the entry point in `ops.cpp` and the op definition in `ggml.h` / `ggml.c`.

## Key Interfaces

```c
// vec.h - FP16/FP32 matrix multiplication and dot product
void ggml_vec_dot_f16(int n, float * s, size_t bs,
                      ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
void ggml_matmul_f16_4x4_kernel(size_t na, size_t nb, int dim,
                                const void * a, const void * b, void * c, size_t c_column_bytes);
void ggml_matmul_f32_4x4_kernel(size_t na, size_t nb, int dim,
                                const void * a, const void * b, void * c, size_t c_column_bytes);
void matmul_outer_8x16_b_micro_packA_f16(size_t M, size_t N, size_t K,
    const void * a, size_t a_column_bytes, const void * b, size_t b_column_bytes,
    void * c, size_t c_column_bytes);

// ggml-cpu-quants.h - Q8_0 MMLA spack packing and computation
void matmul_q8_0_mmla_spack_padded(bool isA, size_t nr_real, size_t nr_pack, int dim,
    const block_q8_0 * X, size_t column_bytes, matmul_q8_0_mmla_spack_t ** out);
void matmul_q8_0_mmla_4x4_tiled_padded(size_t na_real, size_t nb_real,
    size_t na_pad, size_t nb_pad, int K, size_t offM, size_t offN, size_t offK,
    const block_q8_0 * A, const block_q8_0 * B, size_t a_stride, size_t b_stride,
    size_t a_col_bytes, size_t b_col_bytes, float * tmp_c,
    matmul_q8_0_mmla_spack_t ** pack_a, matmul_q8_0_mmla_spack_t ** pack_b);

// ggml.h / ops.h - Fused SDPA
struct ggml_tensor * ggml_fused_cpp_sdpa_ext(struct ggml_context * ctx,
    struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v,
    struct ggml_tensor * mask, float scale);
void ggml_compute_forward_fused_cpp_sdpa_ext(const struct ggml_compute_params * params,
    struct ggml_tensor * dst);
```

## Baseline Fixes

The patches also include functional fixes to the baseline source to ensure fair accuracy verification:

- `common_embd_normalize` / `common_embd_similarity_cos` use double-precision accumulation to avoid FP32 overflow on long sequences;
- `common` links OpenMP when `GGML_OPENMP` is enabled (required by `omp simd`);
- `tools/server` fixes trailing whitespace in the embedding output path and adds a `reserve()` helper for embedding buffers.

## Build and Runtime Helpers

### Compile options

```bash
-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
CFLAGS+=-DGGML_USE_FUSED_CPP_SDPA

-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
```

### One-click build

```bash
./compile.sh build-delivery
```

### Environment variable control

| Environment Variable | Value | Effect |
| --- | --- | --- |
| `GGML_FUSED_CPP_SDPA` | `0`/`off`/`false`/`no` | Disable fused SDPA and fall back to native flash_attn |
| `GGML_FUSED_CPP_SDPA` | `debug`/`trace` | Enable fused SDPA debug logs |
| `GGML_FUSED_CPP_SDPA` | Empty/other | Enabled by default |

## Acceptance Criteria

1. Functional correctness: the output of each optimized kernel has a cosine similarity greater than 0.999 to the native model output on a specified test dataset, or the average score drop across all C-MTEB datasets is less than 1%, verified by the accuracy script.
2. Performance: after optimization, each kernel reaches the expected end-to-end speedup in benchmarks (per-platform performance data is provided in the technical report).
