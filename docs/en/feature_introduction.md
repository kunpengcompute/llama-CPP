# Feature Introduction

The llama-CPP baseline is based on the official llama.cpp commit `3ac67535c86`. This optimization targets the Kunpeng 920B (7280Z) processor and focuses on the ARM matrix multiplication and attention (SDPA) operators in the CPU backend, covering FP16, FP32, and Q8_0 matrix multiplication as well as fused attention computation. It does not modify the upper-layer APIs or the compute graph structure; instead, it replaces the underlying kernels through the `type_traits` callback registration mechanism of the GGML CPU backend, remaining fully compatible with the non-optimized path.

The optimization can be divided into four parts: matrix multiplication, fused attention, baseline fixes, and build/runtime helpers. The matrix multiplication part leverages ARM NEON / SVE-256 / i8mm (MMLA) instructions; the fused attention part merges the FlashAttention v2 flow into a single CPU operator to reduce intermediate tensor traffic.

## FP16 Matrix Multiplication Optimization

When model weights are F16, matrix multiplication mainly occurs in the prefill phase and is the most frequent computation pattern. Four levels of kernels are provided to cover different M and N configurations.

**vec_dot_f16 (NEON software-pipelined dot product)**

Eight `float16x8_t` accumulators prefetch the first 64 elements, and the main loop unrolls in steps of 64 elements, computing the current round while prefetching the next, forming a software pipeline. It performs horizontal accumulation and converts to FP32 at the end as the general dot-product fallback.

**FP16 4×4 tile kernel (NEON outer product)**

Using a 4×4 subblock as the minimum compute unit, it uses a per-row fast path when N==1, and executes 16 `vfmaq_f16` outer-product accumulations per 4×4 tile in the main loop (fp16×fp16→fp32). The M-tail is batch processed and the N-tail falls back to `vec_dot_f16`. It suits small prefill batches (M<16).

**FP16 outer-packA 8×16 kernel (hand-written NEON outer product, large prefill batches)**

A is prepacked (packA) in 8×16 tiles, K is double-buffered in steps of 32, and each lane of every B vector is broadcast-multiplied against 16 A vectors, giving 128 `fmla` outer products (256 total for K step=32). This fully exploits registers and instruction-level parallelism. Constraints are M%16==0 and N%8==0, suiting large-batch prefill (M>=16).

## FP32 Matrix Multiplication Optimization

An SVE 4×4 tile kernel is implemented using the 8-lane FP32 parallelism of 256-bit SVE. It initializes 16 `svfloat32_t` accumulators, chunks K by vector length, performs 16 `svmla_f32_m` operations, then horizontally sums into C. The tail falls back to `ggml_vec_dot_f32`.

## Q8_0 Quantized Matrix Multiplication Optimization (MMLA + Spack)

The Q8_0 optimization delivers the largest performance gain of this work. The core ideas are:

- Repack (spack) the `block_q8_0` data of matrices A and B into the memory layout required by the MMLA (`smmla`, 8-bit×8-bit→32-bit) instruction;
- Each Q8_0 block contains 32 int8 quantized values and one fp16 scale; the micro-kernel uses SMMLA to compute the int8 dot product into int32, then multiplies by the A/B scales and accumulates into FP32;
- Tiling strategy: outer tile M=128 × N=64, K tile=2048, micro tile 8×8;
- A panels are packed in groups of 2 rows, B panels in groups of 4 rows; the same A panel is reused across multiple N tiles to avoid repeated packing.

`type_traits[GGML_TYPE_Q8_0]` registers the `matmul_q8_0_mmla_spack_b_g` callback, falling back to the generic kernel when rows/columns are not tile-aligned.

## Fused SDPA (FlashAttention v2 NEON Fused Operator)

The native llama.cpp attention executes QKᵀ, scaling, mask, softmax, and PV as separate operators, repeatedly reading and writing the large intermediate scores matrix. On the CPU this easily shifts the workload from a compute bottleneck to an L2/L3/memory-bandwidth bottleneck. Fused SDPA merges these steps into a single dedicated C++/NEON/SVE kernel that streams by tile, reducing intermediate tensors and memory traffic.

**Integration**: controlled by the `GGML_USE_FUSED_CPP_SDPA` compile flag. `build_attn_mha()` in `llama-graph.cpp` detects the enabling conditions (ARM64; q/k/v all F32; MHA with n_head==n_head_kv; no MLA/ALiBi/attn_soft_cap; mask is F16/F32 contiguous or absent; v->ne[0]%8==0). When satisfied, it builds the `GGML_OP_FUSED_CPP_SDPA_EXT` op to replace the native flash_attn.

**Diagonal-block optimization**: when `L == S` (prefill) and `L % 64 == 0`, the attention matrix is divided along the diagonal into 64×64 blocks, computing only the upper-triangular blocks inside the causal mask to reduce work.

**NEON kernel**: Q/K/V are packed and transposed inside the micro-kernel, using an L3-cache-aware KV tiling strategy, FP32 internal precision, and pb16 approximate softmax. Three variants are provided (mask_f16 / mask_f32 / maskless) with head-level thread partitioning for parallelism.

## Baseline Fixes

The patches also include functional fixes to the baseline source to ensure fair accuracy verification:

- `common_embd_normalize` / `common_embd_similarity_cos` use double-precision accumulation to avoid FP32 overflow on long sequences;
- `common` links OpenMP when `GGML_OPENMP` is enabled (required by `omp simd`);
- `tools/server` fixes trailing whitespace in the embedding output path and adds a `reserve()` helper for embedding buffers.

## Build and Runtime Helpers

- `compile.sh`: one-click ARM build script. The `build-fused-sdpa` profile uses `-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops` with the fused SDPA path enabled; the default profile is `-O3 RelWithDebInfo`, building `ggml-cpu`, `llama-embedding`, `llama-bench`, and `llama-server` via ninja.
- The `GGML_FUSED_CPP_SDPA` environment variable controls the fused SDPA path (`0`/`off`/`false`/`no` disables it, `debug`/`trace` enables debug logs, empty/other values enable it by default).

For a summary of the kernel algorithms and interfaces, see the Chinese design summary at `docs/zh/设计摘要.md` inside this repository.
