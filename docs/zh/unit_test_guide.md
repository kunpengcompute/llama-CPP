# 算子优化单元测试指南

## 1.测试介绍

`tests/test-sdpa-f16q80-opt`是基于合成（mock）数据的正确性单元测试，用于验证两条推理路径。

| 路径 | ggml 算子 |
| ------ | ----------- |
| 融合注意力（SDPA） | `GGML_OP_FUSED_CPP_SDPA_EXT`（`ggml_fused_cpp_sdpa_ext`） |
| 矩阵乘（GEMM） | `ggml_mul_mat`（F32/F16/Q8_0权重） |

测试不依赖模型文件或任何外部输入：进程内生成合成张量，经真实ggml CPU计算图执行端到端推理，并与C++数学参考实现比对。

## 2. 测试覆盖范围

### 2.1 fused SDPA（6例）

参数`(B, H, L, S, D, DV, scale, mask)`。覆盖多batch、多头、带/不带mask；`DV` 均为8的倍数（算子约束）。

```text
B=1 H=2 L=8  S=8  D=32  DV=32  scale=0.125  mask=0
B=1 H=2 L=8  S=8  D=32  DV=32  scale=0.125  mask=1
B=2 H=4 L=6  S=10 D=64  DV=64  scale=0.088  mask=0
B=2 H=4 L=6  S=10 D=64  DV=64  scale=0.088  mask=1
B=1 H=1 L=16 S=16 D=32  DV=32  scale=0.125  mask=0
B=1 H=1 L=16 S=16 D=32  DV=32  scale=0.125  mask=1
```

### 2.2 GEMM（14例）

**GEMM shape取自真实embedding模型的线性层。**

- **bge-small-zh-v1.5**：`hidden=512, intermediate=2048` -&gt; attn Q/K/V/O `M=512,K=512`，ffn_up `M=2048,K=512`，ffn_down `M=512,K=2048`
- **bge-m3**：`hidden=1024, intermediate=4096` -&gt; attn `M=1024,K=1024`，ffn_up `M=4096,K=1024`，ffn_down `M=1024,K=4096`

```text
权重类型  M    N    K    对应层（模型）
f32     512    8  512  attn（bge-small）
f16     512    8  512  attn（bge-small）-> FP16 NEON GEMM
q8_0    512    8  512  attn（bge-small）-> Q8_0 NEON GEMM
f16    2048    8  512  ffn_up（bge-small）-> FP16 NEON GEMM
q8_0   2048    8  512  ffn_up（bge-small）-> Q8_0 NEON GEMM
f16     512    8 2048  ffn_down（bge-small）-> FP16 NEON GEMM
q8_0    512    8 2048  ffn_down（bge-small）-> Q8_0 NEON GEMM
f32    1024    8 1024  attn（bge-m3）
f16    1024    8 1024  attn（bge-m3）-> FP16 NEON GEMM
q8_0   1024    8 1024  attn（bge-m3）-> Q8_0 NEON GEMM
f16    4096    8 1024  ffn_up（bge-m3）-> FP16 NEON GEMM
q8_0   4096    8 1024  ffn_up（bge-m3）-> Q8_0 NEON GEMM
f16    1024    8 4096  ffn_down（bge-m3）-> FP16 NEON GEMM
q8_0   1024    8 4096  ffn_down（bge-m3）-> Q8_0 NEON GEMM
```

### 2.3 测试涉及的关键路径函数

测试通过真实ggml`mul_mat`/融合SDPA dispatch触发以下优化内核（ARM/NEON），确保**FP16 GEMM NEON 算子**与 **Q8_0 GEMM NEON 算子**被实际执行并验证。

| 路径 | 涉及函数 |
| ------ | ---------- |
| F16 GEMM（NEON） | `matmul_outer_packA_b_g_b16` -&gt; `matmul_outer_8x16_b_micro_packA_f16` / `matmul_outer_4x16_b_micro_packA_f16`；leftover走`ggml_matmul_f16_4x4_kernel` |
| Q8_0 GEMM（NEON） | `matmul_q8_0_mmla_spack_b_g` -&gt; `matmul_q8_0_mmla_8x8_b_micro_spack_neon` / `matmul_q8_0_mmla_4x4_b_micro_spack_neon` |
| F32 GEMM | `ggml_matmul_f32_4x4_kernel` |
| 融合 SDPA | `ggml_compute_forward_fused_cpp_sdpa_ext` -&gt; `fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp` / `..._mask_f16` / `..._mask_f32` |

> **说明**：`f16`与`q8_0`会命中上述NEON GEMM加速内核；`f32`作为参考基线对照。

## 3. 测试原理

测试用带种子的伪随机函数生成可复现的合成数据，将数据写入ggml张量，构建计算图并在CPU后端执行。再由C++参考实现计算期望结果，最后以归一化均方误差（NMSE，`nmse(a, b) = Σ(a - b)² / Σa²`）判定结果是否正确。判定阈值如下：fused SDPA `NMSE < 1e-4`；GEMM按权重类型区分，F32/Q8_0 `NMSE < 1e-4`，F16 `NMSE < 5e-3`（FP16累加精度随K增大而降低，故F16阈值放宽）。

### 3.1 SDPA参考

1. `reference_sdpa()`复现`out = softmax(scale · Q·Kᵀ + mask) · V`。
2. q/k/v/out按ggml张量`ne`的连续内存索引访问（innermost=`ne[0]`）。
3. mask按`[L, S]`布局（`mask[l*S + s]`）叠加。

### 3.2 GEMM参考

1. `reference_matmul()`复现`ggml_mul_mat`约定。

   ```bash
   out[i][j] = Σ_k w[i][k] · act[j][k]
   ```

2. 激活按N×K读取（右侧操作数按转置处理）；输出采用`ne=[M, N]`的raw布局`i + j·M`。

## 4. 编译

fused SDPA算子仅在ARM构建（`GGML_USE_FUSED_CPP_SDPA`）编译进ggml-cpu，测试因此在 `tests/CMakeLists.txt`中以架构门控（`aarch64|arm`）参与构建，随tests一起编译。无需额外CMake变量或模型路径。

```bash
cmake -B build-delivery -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON -DBUILD_TESTING=ON

cmake --build build-delivery --target test-sdpa-f16q80-opt -j
cmake --build build-delivery -j
```

> **说明**：在已有构建目录中新增测试后，需重新执行`cmake .`使其纳入构建系统。

## 5. 启动测试

```bash
build-delivery/bin/test-sdpa-f16q80-opt

ctest --test-dir build-delivery -L fused-sdpa
ctest --test-dir build-delivery -R test-sdpa-f16q80-opt
```

> **说明**：全部用例通过返回0，任一失败返回1。

## 6. 测试预期输出

- build-delivery/bin/test-sdpa-f16q80-opt

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

- ctest --test-dir build-delivery -L fused-sdpa

```text
    Start 29: test-sdpa-f16q80-opt
1/1 Test #29: test-sdpa-f16q80-opt .............   Passed    0.59 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
fused-sdpa    =   0.59 sec*proc (1 test)

Total Test time (real) =   0.60 sec
```

- ctest --test-dir build-delivery -R test-sdpa-f16q80-opt

```text
    Start 29: test-sdpa-f16q80-opt
1/1 Test #29: test-sdpa-f16q80-opt .............   Passed    0.62 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
fused-sdpa    =   0.62 sec*proc (1 test)

Total Test time (real) =   0.62 sec
```

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | 第一次正式发布。<br>- 将测试原理的有序描述调整为段落描述，避免被误认为操作步骤。<br>- 将特殊箭头符号替换为`-&gt;`。 |
