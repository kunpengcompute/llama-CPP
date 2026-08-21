# 版本说明书

## 版本配套说明

### 产品版本信息

| 产品名称 | 产品版本 | 软件名称 | 软件包版本 |
| :--- | :--- | :--- | :--- |
| Kunpeng BoostKit | 26.2.RC1 | llama-CPP | V1.0.0 |

### 与操作系统、编译器和 CPU 配套说明

| 操作系统 | 编译器 | CPU 类型 | 基线版本 |
| :--- | :--- | :--- | :--- |
| openEuler 22.03 LTS SP3 | GCC/G++ 15.2.0 或更新版本 | 鲲鹏920B/鲲鹏950 | llama.cpp commit `3ac67535c86` |

## V1.0.0

### 更新说明

llama-CPP是针对鲲鹏处理器进行的CPU算子性能优化补丁合集，聚焦llama.cpp推理引擎CPU后端的ARM矩阵乘法与注意力算子，采用ARM NEON/SVE-256/i8mm指令集实现鲲鹏亲和优化，提升矩阵乘法与注意力计算性能。

主要优化以下能力。

- FP16矩阵乘法优化：NEON软件流水线点积、4×4外积tile、outer-packA 8×16手写外积kernel。
- FP32矩阵乘法优化：SVE 4×4 tile kernel。
- Q8_0量化矩阵乘法优化：MMLA（i8mm+Spack数据打包。
- Fused SDPA（FlashAttention v2 NEON融合算子）与对角分块优化。
- 基线embedding normalize/similarity双精度修复及server输出路径相关修复。

### 已解决的问题

解决FP16、Q80算子在鲲鹏CPU上性能差的问题。

### 遗留问题

无

## 版本配套文档

### V1.0.0版本配套文档

| 文档名称 | 内容简介 | 交付方式 |
| :--- | :--- | :--- |
| 版本说明书 | 提供llama-CPP每个发布版本的基础信息和特性更新信息。 | 开源仓 |
| 用户指南 | 提供llama-CPP优化使用说明。 | 开源仓 |
| 特性介绍 | 提供llama-CPP优化说明。 | 开源仓 |
| 技术报告 | 提供精度与性能验证数据。 | 开源仓 |

### 获取文档的方法

您可以通过访问[开源仓](https://gitcode.com/boostkit/llama-CPP/blob/opt/docs)浏览和获取相关文档。
