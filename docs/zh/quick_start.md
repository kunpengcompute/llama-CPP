# 快速入门

本文介绍如何基于官方llama.cpp（commit `3ac67535c86`）checkout代码、合入优化补丁并完成编译，快速体验llama-CPP在鲲鹏950/鲲鹏920B上提供的CPU算子优化能力。优化补丁聚焦FP16/FP32/Q8_0矩阵乘法与fused SDPA注意力算子，可提升llama.cpp在CPU小模型推理场景下的性能。

## 环境要求

- 硬件平台：鲲鹏950/鲲鹏920B（NEON / SVE-256 / i8mm）
- 操作系统：openEuler 22.03 LTS SP3
- 编译器：GCC/G++ 15.2.0或更新版本（需支持`armv8.6-a+dotprod+i8mm+sve`）
- 构建依赖：cmake >= 3.20、ninja（推荐）、git

## 1. 获取代码并checkout到对应commit

优化补丁基于官方llama.cpp（commit `3ac67535c86`），请先将源码切换并固定到该commit。

1. 拉取官方llama.cpp源码。

   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   ```

2. checkout到官方llama.cpp（commit `3ac67535c86`）对应版本。

   ```bash
   git checkout 3ac67535c86
   ```

## 2. 合入优化补丁

llama-CPP优化补丁位于仓库`patch/`目录下的`0001`至`0011`补丁文件（`00*.patch`），在llama.cpp源码目录中合并即可。

1. 获取llama-CPP仓库（如还未获取）。

   ```bash
   git clone <llama-CPP 仓库地址>
   cd <llama-CPP 仓库路径>
   ```

2. 先运行检查，确认补丁可以正常应用。

   ```bash
   git apply --check /path/to/llama-CPP/patch/00*.patch
   ```

3. 正式合入优化补丁。

   ```bash
   git am /path/to/llama-CPP/patch/00*.patch
   ```

应用时会出现trailing whitespace警告，这是源码本身带的行尾空格（如`server.cpp`、`quants.h`），不影响结果。

## 3. 编译

使用`compile.sh`一键编译（`compile.sh`由补丁`0009`、`0010`一并引入）。

```bash
./compile.sh build-delivery
```

>**说明：**
>默认编译（-O3 RelWithDebInfo）会构建ggml-cpu、llama-embedding、llama-bench和llama-server，编译结果位于`build-delivery/bin/`。

## 4. 验证与进一步阅读

编译完成后，可运行单元测试验证优化算子正确性。

```bash
build-delivery/bin/test-sdpa-f16q80-opt
```

更多内容请参见：

- 《[用户指南](./user_guide.md)》：bge-m3等Embedding模型的部署与调优。
- 《[特性介绍](./feature_introduction.md)》：优化算子的特性与原理。
- 《[单元测试指南](./unit_test_guide.md)》：单元测试的编译、执行与预期输出。
- 《[技术报告](./technical_report.md)》：精度与性能验证数据。
- 《[版本说明书](./release_notes.md)》：发布版本信息与配套文档。

## 修订记录

| 发布日期 | 修订记录 |
| :--- | :--- |
| 2026-08-24 | 第一次正式发布。<br>- 新增本文档，提供基于官方llama.cpp（commit `3ac67535c86`）checkout代码、合入优化补丁与编译的快速入门指引。 |
