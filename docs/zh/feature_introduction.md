# 特性介绍

## 特性描述

llama-CPP优化补丁基于官方llama.cpp（commit `3ac67535c86`）。本优化面向**鲲鹏950处理器/鲲鹏920新型号处理器**，利用ARM NEON/SVE-256/i8mm指令集，聚焦llama.cpp推理引擎CPU后端的矩阵乘法与注意力（SDPA）算子，覆盖FP16、FP32、Q8_0三种数据类型的矩阵乘法以及融合注意力计算。整体不修改上层API与计算图结构，通过GGML CPU Backend的 `type_traits` 回调注册机制替换底层kernel，保证与非优化路径完全兼容。优化补丁基于llama.cpp的计算图调度机制，在GGML CPU Backend中注入自研高性能kernel，覆盖FP16、FP32、Q8_0三种数据类型的矩阵乘法以及融合SDPA（注意力）运算。

## 架构介绍

| 层次 | 优化状态 | 关键组件 |
| --- | --- | --- |
| 应用层 | 未修改 | llama-embedding / llama-bench / llama-server |
| 框架层 | 部分修改 | GGML OP IR（新增GGML_OP_FUSED_CPP_SDPA_EXT）-&gt; type_traits matmul路由+fused SDPA op分发 |
| Kernel 层 | 全部新增 | FP16/FP32/Q8_0 MatMul+Fused SDPA（NEON/SVE/i8mm） |
| 硬件层 | 未修改 | NEON、SVE-256（256-bit）、i8mm；OpenMP并行；64B对齐内存 |

数据流如下。

1. 应用层经`llama_eval()`进入推理循环，在框架层构建GGML计算图并由 `ggml_graph_compute()`调度。
2. CPU Backend根据`vec_dot_type`选择优化kernel。
3. Q8_0 执行MMLA spack打包和tile计算，各kernel在底层利用SVE-256/NEON/i8mm指令完成 SIMD加速计算。

## FP16矩阵乘法优化

Embedding模型在权重为F16时，矩阵乘法主要发生在prefill阶段，是最频繁的计算模式。为覆盖不同M、N配置，Embedding模型提供四个层次kernel。

### vec_dot_f16（NEON软件流水线点积，通用兜底）

1. 初始化8个float16x8_t累加器 (acc0..acc7)。
2. 预加载前64个元素到ax0..ax7/ay0..ay7。
3. 主循环运算命令如下。

   ```bash
    (i=64; i<np; i+=64):
     acc[k] += ax[k] * ay[k]        
     ax[k] = vld1q_f16(&x[i+k*8])   
     ay[k] = vld1q_f16(&y[i+k*8])
    ```

4. 尾循环处理最后64元素。
5. vpaddq_f16水平累加+ vcvt_f32_f16转f32求和。

### FP16 4×4 tile kernel（NEON外积）

1. N==1 fast path：按M行循环，每行一个vfmaq_f16累加，避免函数调用开销。
2. 4×4 tile主循环：
   - 加载`A[j..j+3][k..k+8]`、`B[i..i+3][k..k+8]`。
   - 16次vfmaq_f16外积累加（j0×i0..i3, j1×i0..i3, ...）。
   - 水平累加vcvt_f32_f16 + vaddvq_f32写入C。
3. M尾循环（1-3 行）批量处理；N尾循环回退vec_dot_f16。

>**说明：**
>FP16 4×4 tile kernel（NEON外积）适用于prefill小批量（M<16）。

### FP16 outer-packA 8×16 kernel（手写NEON外积，prefill大批量）

输入：`packA(预打包A)[M_packed, K]`、`B[N,K]`（fp16）；约束设置为 `M%16==0`、`N%8==0`。

1. 主循环 (N按8, M按16)：初始化16个NEON累加器 (v12..v27)。
2. K主循环(step=32, 双block)：
   - 加载16×2个A vector（从packA中）、8个B vector。
   - 每个B向量lane广播乘16个A vector -&gt; 8×16=128次fmla（step=32共256次）。
3. K全块尾循环(step=16)；K标量尾循环(step=2)。
4. fcvtl f16-&gt;f32写回C。
5. 实现结果位于`ggml/src/ggml-cpu/vec.cpp`/`vec.h`。

>**说明：**
>将A按8×16 tile预打包（packA）并充分使用寄存器与指令级并行，适用于M>=16的大批量 prefill场景。

## FP32矩阵乘法优化

对FP32矩阵乘法实现SVE 4×4 tile kernel，利用256-bit SVE的8-lane FP32并行能力，一次性计算4×4=16个点积。

1. N/M按4对齐主循环：初始化16个svfloat32_t累加器（`acc[0..3][0..3]`）。
2. K按vl（向量长度）分块：
   - 加载`A[j+0..3][k..k+vl]`到j0..j3。
   - 加载`B[i+0..3][k..k+vl]`到i0..i3。
   - `acc[m][n] += j[m] * i[n]`（16次svmla_f32_m）。
3. svaddv水平求和写入C；尾循环回退ggml_vec_dot_f32。
4. 实现结果位于`ggml/src/ggml-cpu/vec.cpp`/`vec.h`。

## Q8_0量化矩阵乘法优化（MMLA+Spack）

Q8_0优化是性能提升最大的模块。核心思路是将A、B矩阵的`block_q8_0`数据按MMLA（i8mm `smmla`，8bit×8bit-&gt;32bit）指令所需内存布局重新打包（spack）。micro-kernel用SMMLA计算int8点积得到int32，再乘A/B scale累加进FP32。

Q8_0每个block包含32个int8量化值与1个fp16 scale，两个block做点积，命令如下。

``` bash
Σk (da×qa[k]) × (db×qb[k]) = (da×db) × Σk qa[k]×qb[k]

int32_sum = Σ qa × qb            # SMMLA 计算
float_sum = int32_sum × scale_a × scale_b
C        += float_sum
```

### 打包布局（Spack）

- A按2行一组打包：每两行A的一个Q8_0 block交错打包为+0/+16/+32/+48四段。每段16B前8B 为第0行的8个int8，后8B为第1行的8个int8。正好构成SMMLA需要的2行×8个K输入。
- B按4行一组打包：每四行B的一个block打包为+0/+16/+32/+48/+64/+80/+96/+112八段，对应cols0/1、cols2/3与K0..7、8..15、16..23、24..31的组合。
- 同一个A panel在多个N tile间复用，不再针对每个N tile重复打包A。

### 分块策略

Outer tile M=128×N=64，K tile=2048，Micro tile 8×8。完整逻辑如下。

完整Q8_0 MatMul的执行流程如下。

1. 清零C。
2. 按M分块：每次最多处理128行A。
3. 按K分块：每次最多处理2048个K。
4. 打包A panel（按2行/组）。
5. 按N分块：每次最多处理6行B。
6. 打包B panel（按4行/组）。
7. 执行micro-kernel（`matmul_q8_0_mmla_8x8_b_micro_spack_neon`）：
   - 每次处理一个8×8 C tile（拆成上/下半4×8）。
   - 每个Q8_0 block依次对K0..7/K8..15/K16..23/K24..31执行SMMLA，再按int32转fp32、乘以A scale、乘以B scale的顺序累加到C。

Q8_0 MatMul的调用流程如下。

1. `ggml_compute_forward_mul_mat()`调用`compute_forward_mul_mat_one_chunk_matmul_kernel()`。
2. 该函数通过`type_traits[Q8_0].matmul`路由到`matmul_q8_0_mmla_spack_b_g()`回调。
3. `matmul_q8_0_mmla_spack_b()`执行外层M/N/K分块并打包A/B panel。
4. 最终调用`matmul_q8_0_mmla_8x8_b_micro_spack_neon()`完成微内核计算。

>**说明：**
>`type_traits[GGML_TYPE_Q8_0]`注册`matmul_q8_0_mmla_spack_b_g`回调，行/列不满足 tile对齐时回退到通用kernel。实现位于`ggml/src/ggml-cpu/ggml-cpu-quants.c/.h`，路由注册在`ggml-cpu.c`。

## Fused SDPA（FlashAttention v2 NEON融合算子）

官方llama.cpp Attention由QKᵀ、缩放、Mask、Softmax、PV等多个独立算子依次完成。中间 scores矩阵被反复读写，CPU上易从“算力瓶颈”变为“L2/L3/内存带宽瓶颈”。Fused SDPA将这些步骤融合进一个专用C++/NEON/SVE kernel，按tile流式完成，减少中间张量与内存流量。

### 集成方式

通过`GGML_USE_FUSED_CPP_SDPA`编译宏控制。`llama-graph.cpp`的`build_attn_mha()`检测启用条件如下。

- ARM64平台；q/k/v均为F32；MHA（n_head==n_head_kv）。
- 无MLA、无ALiBi、无attn_soft_cap。
- mask为F16/F32 contiguous或无mask；`v->ne[0] % 8 == 0`。

满足条件时构建`GGML_OP_FUSED_CPP_SDPA_EXT`op替代官方flash_attn。

### 推理流程

推理流程如下。

1. `build_attn_mha()`判断是否使用fused SDPA路径。
2. 当满足启用条件时，构建`GGML_OP_FUSED_CPP_SDPA_EXT`算子并调用`ggml_fused_cpp_sdpa_ext(q, k, v, mask, scale)`，经`ops.cpp`中的`ggml_compute_forward_fused_cpp_sdpa_ext()`执行以下内容。
   - 当L等于S且L能被64整除时，采用对角分块（64×64 blocks，只计算因果上三角）。
   - 否则，全矩阵调用。`fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp()`。
3. 当不满足启用条件时，回退到官方flash_attn（flash_attn_ext）。

### 对角分块优化

当`L == S`（prefill）且`L % 64 == 0`时，把注意力矩阵沿对角线划分为64×64块，只计算因果mask内上三角块，减少计算量。

### NEON kernel

- Q/K/V在微内核内完成。
- pack+transpose，避免额外内存移动。
- 采用L3-cache-aware的KV分块策略。
- fp32内部精度、pb16格式近似softmax。
- 提供mask_f16/mask_f32/无mask三种变体，并按head分区到线程并行。
- 实现位于`ggml/src/ggml-cpu/fused-cpp/fp32_packqkv/`，入口在`ops.cpp`，op定义在 `ggml.h`和`ggml.c`。

## 关键接口

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

## 官方llama.cpp问题修复

优化补丁还包含对官方llama.cpp源码的若干功能性修复，用于保证精度验证的公平性。

- `common_embd_normalize`/`common_embd_similarity_cos`改为双精度累加，避免长序列下FP32溢出。
- `common`在`GGML_OPENMP`开启时链接OpenMP（`omp simd`需要）。
- `tools/server`修复embedding输出路径的行尾空白问题，并新增embedding缓冲，用 `reserve()`辅助函数。

## 编译与运行辅助

### 编译选项

- fused SDPA编译（推荐，用于开启fused SDPA路径）

```bash
-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
CFLAGS+=-DGGML_USE_FUSED_CPP_SDPA
```

- 标准编译（仅矩阵乘法优化，不包含fused SDPA）

```bash
-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops
```

- 一键编译

   ```bash
   ./compile.sh build-delivery
   ```

### 环境变量控制

| 环境变量 | 值 | 效果 |
| --- | --- | --- |
| `GGML_FUSED_CPP_SDPA` | `0`/`off`/`false`/`no` | 关闭fused SDPA，回退到官方flash_attn |
| `GGML_FUSED_CPP_SDPA` | `debug`/`trace` | 启用fused SDPA调试日志 |
| `GGML_FUSED_CPP_SDPA` | 空/其他 | 默认启用 |

## 验收标准

- 功能正确性：优化后各kernel输出与官方llama.cpp输出在指定测试数据集上的余弦相似度大于0.999，或C-MTEB所有数据集平均得分掉点小于1%，且通过脚本实现精度验证。
- 性能：各kernel优化后，在基准测试中端到端性能达成预期加速比（具体平台的性能数据参见《[技术报告](./technical_report.md)》）。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | 第一次正式发布 |
