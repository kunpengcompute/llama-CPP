# 特性介绍

llama-CPP 基线基于官方 llama.cpp 的 commit `3ac67535c86`。本优化面向**鲲鹏处理器**，利用 ARM NEON / SVE-256 / i8mm 指令集，聚焦 llama.cpp 推理引擎 CPU 后端的矩阵乘法与注意力（SDPA）算子，覆盖 FP16、FP32、Q8_0 三种数据类型的矩阵乘法以及融合注意力计算。整体不修改上层 API 与计算图结构，通过 GGML CPU Backend 的 `type_traits` 回调注册机制替换底层 kernel，保证与非优化路径完全兼容。

## 整体架构

优化基于 llama.cpp 的计算图调度机制，在 GGML CPU Backend 中注入自研高性能 kernel，覆盖 FP16、FP32、Q8_0 三种数据类型的矩阵乘法以及融合 SDPA（注意力）运算。

| 层次 | 优化状态 | 关键组件 |
| --- | --- | --- |
| 应用层 | 未修改 | llama-embedding / llama-bench / llama-server |
| 框架层 | 部分修改 | GGML OP IR（新增 GGML_OP_FUSED_CPP_SDPA_EXT）→ type_traits matmul 路由 + fused SDPA op 分发 |
| Kernel 层 | 全部新增 | FP16 / FP32 / Q8_0 MatMul + Fused SDPA（NEON / SVE / i8mm） |
| 硬件层 | 未修改 | NEON、SVE-256（256-bit）、i8mm；OpenMP 并行；64B 对齐内存 |

数据流：应用层经 `llama_eval()` 进入推理循环 → 框架层构建 GGML 计算图并由 `ggml_graph_compute()` 调度 → CPU Backend 根据 `vec_dot_type` 选择优化 kernel（F32/F16 → `ops.cpp` 中的 matmul kernel 或 fused SDPA 入口；Q8_0 → MMLA spack 打包 + tile 计算）→ 各 kernel 在底层利用 SVE-256 / NEON / i8mm 指令完成 SIMD 加速计算。

## FP16 矩阵乘法优化

Embedding 模型在权重为 F16 时，矩阵乘法主要发生在 prefill 阶段，是最频繁的计算模式。为覆盖不同 M、N 配置，提供四个层次 kernel。

**vec_dot_f16（NEON 软件流水线点积，通用兜底）**

```
1. 初始化 8 个 float16x8_t 累加器 (acc0..acc7)
2. 预加载前 64 个元素到 ax0..ax7 / ay0..ay7
3. 主循环 (i=64; i<np; i+=64):
     acc[k] += ax[k] * ay[k]        # 计算本轮
     ax[k] = vld1q_f16(&x[i+k*8])   # 预取下一轮（软件流水线）
     ay[k] = vld1q_f16(&y[i+k*8])
4. 尾循环处理最后 64 元素
5. vpaddq_f16 水平累加 + vcvt_f32_f16 转 f32 求和
```

**FP16 4×4 tile kernel（NEON 外积）**

```
1. N==1 fast path：按 M 行循环，每行一个 vfmaq_f16 累加，避免函数调用开销
2. 4×4 tile 主循环：
   - 加载 A[j..j+3][k..k+8]、B[i..i+3][k..k+8]
   - 16 次 vfmaq_f16 外积累加（j0×i0..i3, j1×i0..i3, ...）
   - 水平累加 vcvt_f32_f16 + vaddvq_f32 写入 C
3. M 尾循环（1-3 行）批量处理；N 尾循环回退 vec_dot_f16
```

适用于 prefill 小批量（M<16）。

**FP16 outer-packA 8×16 kernel（手写 NEON 外积，prefill 大批量）**

```
输入: packA(预打包A)[M_packed, K], B[N,K] (fp16)；约束 M%16==0, N%8==0
1. 主循环 (N 按 8, M 按 16)：初始化 16 个 NEON 累加器 (v12..v27)
2. K 主循环 (step=32, 双 block)：
   - 加载 16×2 个 A vector（从 packA 中）、8 个 B vector
   - 每个 B 向量 lane 广播乘 16 个 A vector → 8 × 16 = 128 次 fmla（step=32 共 256 次）
3. K 全块尾循环 (step=16)；K 标量尾循环 (step=2)
4. fcvtl f16→f32 写回 C
```

将 A 按 8×16 tile 预打包（packA）并充分使用寄存器与指令级并行，适用于 M>=16 的大批量 prefill 场景。

实现位于 `ggml/src/ggml-cpu/vec.cpp` / `vec.h`。

## FP32 矩阵乘法优化

对 FP32 矩阵乘法实现 SVE 4×4 tile kernel，利用 256-bit SVE 的 8-lane FP32 并行能力，一次性计算 4×4=16 个点积：

```
1. N/M 按 4 对齐主循环：初始化 16 个 svfloat32_t 累加器 (acc[0..3][0..3])
2. K 按 vl（向量长度）分块：
   - 加载 A[j+0..3][k..k+vl] 到 j0..j3
   - 加载 B[i+0..3][k..k+vl] 到 i0..i3
   - acc[m][n] += j[m] * i[n]（16 次 svmla_f32_m）
3. svaddv 水平求和写入 C；尾循环回退 ggml_vec_dot_f32
```

实现位于 `ggml/src/ggml-cpu/vec.cpp` / `vec.h`。

## Q8_0 量化矩阵乘法优化（MMLA + Spack）

Q8_0 优化是本工作中性能提升最大的模块。核心思路：将 A、B 矩阵的 `block_q8_0` 数据按 MMLA（i8mm `smmla`，8bit×8bit→32bit）指令所需内存布局重新打包（spack），micro-kernel 用 SMMLA 计算 int8 点积得到 int32，再乘 A/B scale 累加进 FP32。

Q8_0 每个 block 包含 32 个 int8 量化值与 1 个 fp16 scale，两个 block 做点积：

```
Σk (da×qa[k]) × (db×qb[k]) = (da×db) × Σk qa[k]×qb[k]

int32_sum = Σ qa × qb            # SMMLA 计算
float_sum = int32_sum × scale_a × scale_b
C        += float_sum
```

**打包布局（Spack）**

- A 按 2 行一组打包：每两行 A 的一个 Q8_0 block 交错打包为 +0/+16/+32/+48 四段，每段 16B 前 8B 为第 0 行的 8 个 int8、后 8B 为第 1 行的 8 个 int8，正好构成 SMMLA 需要的 2 行×8 个 K 输入；
- B 按 4 行一组打包：每四行 B 的一个 block 打包为 +0/+16/+32/+48/+64/+80/+96/+112 八段，对应 cols0/1、cols2/3 与 K0..7、8..15、16..23、24..31 的组合；
- 同一个 A panel 在多个 N tile 间复用，不再针对每个 N tile 重复打包 A。

**分块策略**

Outer tile M=128 × N=64，K tile=2048，Micro tile 8×8。完整逻辑：

```
完整 Q8_0 MatMul
├─ C 清零
├─ M 分块：每次最多 128 行 A
│  └─ K 分块：每次最多 2048
│     ├─ pack A panel（按 2 行/组）
│     └─ N 分块：每次最多 64 行 B
│        ├─ pack B panel（按 4 行/组）
│        └─ micro-kernel（matmul_q8_0_mmla_8x8_b_micro_spack_neon）
│           ├─ 每次处理一个 8×8 C tile（拆成上/下半 4×8）
│           └─ 每个 Q8_0 block：K0..7 / K8..15 / K16..23 / K24..31 → SMMLA
│              int32→fp32 → ×A scale → ×B scale → 累加到 C
```

调用链：`ggml_compute_forward_mul_mat()` → `compute_forward_mul_mat_one_chunk_matmul_kernel()` → `type_traits[Q8_0].matmul = matmul_q8_0_mmla_spack_b_g()` → `matmul_q8_0_mmla_spack_b()`（外层 M/N/K 分块、pack A/B panel）→ `matmul_q8_0_mmla_8x8_b_micro_spack_neon()`。

`type_traits[GGML_TYPE_Q8_0]` 注册 `matmul_q8_0_mmla_spack_b_g` 回调，行/列不满足 tile 对齐时回退到通用 kernel。实现位于 `ggml/src/ggml-cpu/ggml-cpu-quants.c/.h`，路由注册在 `ggml-cpu.c`。

## Fused SDPA（FlashAttention v2 NEON 融合算子）

原生 llama.cpp Attention 由 QKᵀ、缩放、Mask、Softmax、PV 等多个独立算子依次完成，中间 scores 矩阵被反复读写，CPU 上易从“算力瓶颈”变为“L2/L3/内存带宽瓶颈”。Fused SDPA 将这些步骤融合进一个专用 C++/NEON/SVE kernel，按 tile 流式完成，减少中间张量与内存流量。

**集成方式**：通过 `GGML_USE_FUSED_CPP_SDPA` 编译宏控制，`llama-graph.cpp` 的 `build_attn_mha()` 检测启用条件：

- ARM64 平台；q/k/v 均为 F32；MHA（n_head == n_head_kv）；
- 无 MLA、无 ALiBi、无 attn_soft_cap；
- mask 为 F16/F32 contiguous 或无 mask；v->ne[0] % 8 == 0。

满足条件时构建 `GGML_OP_FUSED_CPP_SDPA_EXT` op 替代原生 flash_attn。

**推理流程**

```
build_attn_mha()
├─ use_fused_cpp_attn?
│  ├─ YES → ggml_fused_cpp_sdpa_ext(q, k, v, mask, scale)
│  │        → ops.cpp: ggml_compute_forward_fused_cpp_sdpa_ext()
│  │           按 head 分区到线程
│  │           ├─ L == S 且 L%64==0 → 对角分块（64×64 blocks，只算因果上三角）
│  │           └─ else → 全矩阵调用
│  │               fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp()
│  └─ NO → 原生 flash_attn（flash_attn_ext）
```

**对角分块优化**：当 `L == S`（prefill）且 `L % 64 == 0` 时，把注意力矩阵沿对角线划分为 64×64 块，只计算因果 mask 内上三角块，减少计算量。

**NEON kernel**：Q/K/V 在微内核内完成 pack+transpose，避免额外内存移动；采用 L3-cache-aware 的 KV 分块策略；fp32 内部精度、pb16 格式近似 softmax；提供 mask_f16 / mask_f32 / 无 mask 三种变体，并按 head 分区到线程并行。

实现位于 `ggml/src/ggml-cpu/fused-cpp/fp32_packqkv/`，入口在 `ops.cpp`，op 定义在 `ggml.h` / `ggml.c`。

## 关键接口

```c
// vec.h —— FP16/FP32 矩阵乘法与点积
void ggml_vec_dot_f16(int n, float * s, size_t bs,
                      ggml_fp16_t * x, size_t bx, ggml_fp16_t * y, size_t by, int nrc);
void ggml_matmul_f16_4x4_kernel(size_t na, size_t nb, int dim,
                                const void * a, const void * b, void * c, size_t c_column_bytes);
void ggml_matmul_f32_4x4_kernel(size_t na, size_t nb, int dim,
                                const void * a, const void * b, void * c, size_t c_column_bytes);
void matmul_outer_8x16_b_micro_packA_f16(size_t M, size_t N, size_t K,
    const void * a, size_t a_column_bytes, const void * b, size_t b_column_bytes,
    void * c, size_t c_column_bytes);

// ggml-cpu-quants.h —— Q8_0 MMLA spack 打包与计算
void matmul_q8_0_mmla_spack_padded(bool isA, size_t nr_real, size_t nr_pack, int dim,
    const block_q8_0 * X, size_t column_bytes, matmul_q8_0_mmla_spack_t ** out);
void matmul_q8_0_mmla_4x4_tiled_padded(size_t na_real, size_t nb_real,
    size_t na_pad, size_t nb_pad, int K, size_t offM, size_t offN, size_t offK,
    const block_q8_0 * A, const block_q8_0 * B, size_t a_stride, size_t b_stride,
    size_t a_col_bytes, size_t b_col_bytes, float * tmp_c,
    matmul_q8_0_mmla_spack_t ** pack_a, matmul_q8_0_mmla_spack_t ** pack_b);

// ggml.h / ops.h —— Fused SDPA
struct ggml_tensor * ggml_fused_cpp_sdpa_ext(struct ggml_context * ctx,
    struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v,
    struct ggml_tensor * mask, float scale);
void ggml_compute_forward_fused_cpp_sdpa_ext(const struct ggml_compute_params * params,
    struct ggml_tensor * dst);
```

## 基线问题修复

补丁还包含对基线源码的若干功能性修复，用于保证精度验证的公平性：

- `common_embd_normalize` / `common_embd_similarity_cos` 改为双精度累加，避免长序列下 FP32 溢出；
- `common` 在 `GGML_OPENMP` 开启时链接 OpenMP（`omp simd` 需要）；
- `tools/server` 修复 embedding 输出路径的行尾空白问题，并新增 embedding 缓冲用 `reserve()` 辅助函数。

## 编译与运行辅助

**编译选项**

```
# fused-sdpa 编译（推荐，开启 fused SDPA 路径）
-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
CFLAGS+=-DGGML_USE_FUSED_CPP_SDPA

# 标准编译（仅矩阵乘法优化，不包含 fused SDPA）
-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
```

**一键编译**

```bash
# 默认编译（-O3 RelWithDebInfo，构建 ggml-cpu / llama-embedding / llama-bench / llama-server）
./compile.sh build-delivery

# 开启 fused SDPA 的 ARM 编译
./compile.sh build-fused-sdpa
```

**环境变量控制**

| 环境变量 | 值 | 效果 |
| --- | --- | --- |
| `GGML_FUSED_CPP_SDPA` | `0`/`off`/`false`/`no` | 关闭 fused SDPA，回退到原生 flash_attn |
| `GGML_FUSED_CPP_SDPA` | `debug`/`trace` | 启用 fused SDPA 调试日志 |
| `GGML_FUSED_CPP_SDPA` | 空/其他 | 默认启用 |

## 验收标准

1. 功能正确性：优化后各 kernel 输出与原生模型输出在指定测试数据集上余弦相似度大于 0.999，或 C-MTEB 所有数据集平均得分掉点小于 1%，且通过精度验证脚本验证。
2. 性能：各 kernel 优化后，在基准测试中端到端性能达成预期加速比（具体平台的性能数据参见《[技术报告](./technical_report.md)》）。
