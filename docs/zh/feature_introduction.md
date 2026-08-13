# 特性介绍

llama-CPP 基线基于官方 llama.cpp 的 commit `3ac67535c86`。本优化面向鲲鹏处理器，聚焦 CPU 后端的 ARM 矩阵乘法与注意力（SDPA）算子，覆盖 FP16、FP32、Q8_0 三种数据类型的矩阵乘法以及融合注意力计算。整体不修改上层 API 与计算图结构，通过 GGML CPU Backend 的 `type_traits` 回调注册机制替换底层 kernel，保证与非优化路径完全兼容。

优化可划分为矩阵乘法、融合注意力、基线修复与编译运行辅助四大部分。矩阵乘法部分依托 ARM NEON / SVE-256 / i8mm（MMLA）指令集实现鲲鹏亲和优化；融合注意力部分将 FlashAttention v2 流程合并进单一 CPU 算子，减少中间张量搬运。

## FP16 矩阵乘法优化

Embedding 模型在权重为 F16 时，矩阵乘法主要发生在 prefill 阶段，是最频繁的计算模式。为覆盖不同 M、N 配置，提供四个层次的 kernel。

**vec_dot_f16（NEON 软件流水线点积）**

将 8 个 `float16x8_t` 累加器预加载前 64 个元素，主循环按 64 元素步长展开，计算当前轮同时预取下一轮数据，形成软件流水线；末尾做水平累加并转 FP32 求和，作为通用点积兜底路径。

**FP16 4×4 tile kernel（NEON 外积）**

以 4×4 子块为最小计算单元，N==1 时走按行累加的 fast path，主循环对 4×4 tile 执行 16 次 `vfmaq_f16` 外积累加（fp16×fp16→fp32），M 尾循环批量处理、N 尾循环回退 `vec_dot_f16`。适用于 prefill 小批量（M<16）。

**FP16 outer-packA 8×16 kernel（手写 NEON 外积，prefill 大批量）**

将 A 按 8×16 tile 预打包（packA），K 按 32 展开双缓冲，每个 B 向量 lane 广播乘 16 个 A 向量，合计 128 次 `fmla` 外积（K step=32 时共 256 次），充分利用寄存器与指令级并行。约束 M%16==0、N%8==0，适用于 M>=16 的大批量 prefill 场景。

## FP32 矩阵乘法优化

对 FP32 矩阵乘法实现 SVE 4×4 tile kernel，利用 256-bit SVE 的 8-lane FP32 并行能力，初始化 16 个 `svfloat32_t` 累加器，按向量长度对 K 分块，执行 16 次 `svmla_f32_m` 后水平求和写入 C；尾循环回退 `ggml_vec_dot_f32`。

## Q8_0 量化矩阵乘法优化（MMLA + Spack）

Q8_0 优化是本工作中性能提升最大的模块。核心思路为：

- 将 A、B 矩阵的 `block_q8_0` 数据按 MMLA（i8mm `smmla`，8-bit×8-bit→32-bit）指令所需内存布局重新打包（spack）；
- 每个 Q8_0 block 包含 32 个 int8 量化值与 1 个 fp16 scale，micro-kernel 用 SMMLA 计算 int8 点积得到 int32，再乘 A/B scale 累加到 FP32；
- 分块策略：Outer tile M=128 × N=64，K tile=2048，Micro tile 8×8；
- A panel 按 2 行一组打包，B panel 按 4 行一组打包，同一个 A panel 在多个 N tile 间复用，避免重复打包。

`type_traits[GGML_TYPE_Q8_0]` 注册 `matmul_q8_0_mmla_spack_b_g` 回调，行/列不满足 tile 对齐时回退到通用 kernel。

## Fused SDPA（FlashAttention v2 NEON 融合算子）

原生 llama.cpp Attention 由 QKᵀ、缩放、Mask、Softmax、PV 等多个独立算子依次完成，中间 scores 矩阵会被反复读写，CPU 上易从“算力瓶颈”变为“L2/L3/内存带宽瓶颈”。Fused SDPA 将上述步骤融合进一个专用 C++/NEON/SVE kernel，按 tile 流式完成，减少中间张量与内存流量。

**集成方式**：通过 `GGML_USE_FUSED_CPP_SDPA` 编译宏控制，`llama-graph.cpp` 的 `build_attn_mha()` 检测启用条件（ARM64、q/k/v 均为 F32、MHA(n_head==n_head_kv)、无 MLA/ALiBi/attn_soft_cap、mask 为 F16/F32 contiguous 或无 mask、v->ne[0]%8==0），满足时构建 `GGML_OP_FUSED_CPP_SDPA_EXT` op 替代原生 flash_attn。

**对角分块优化**：当 `L == S`（prefill）且 `L % 64 == 0` 时，将注意力矩阵沿对角线划分为 64×64 块，只计算因果 mask 内上三角块，减少计算量。

**NEON kernel**：Q/K/V 在微内核内完成 pack+transpose，采用 L3-cache-aware 的 KV 分块策略，fp32 内部精度、pb16 格式近似 softmax；提供 mask_f16 / mask_f32 / 无 mask 三种变体，并按 head 分区到线程并行。


## 编译与运行辅助

- `compile.sh`：一键 ARM 编译脚本。`build-fused-sdpa` 配置使用 `-march=armv8.6-a+dotprod+i8mm+sve -O3 -funroll-loops` 并开启 fused SDPA 路径；默认配置为 `-O3 RelWithDebInfo`，通过 ninja 构建 `ggml-cpu`、`llama-embedding`、`llama-bench` 与 `llama-server`。
- `GGML_FUSED_CPP_SDPA` 环境变量控制 fused SDPA 启停（`0`/`off`/`false`/`no` 关闭，`debug`/`trace` 开启调试日志，空/其他默认启用）。

详细的 kernel 算法与接口设计，请参见《[设计摘要](./设计摘要.md)》。
