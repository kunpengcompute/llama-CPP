# llama-CPP介绍

## 最新消息

- [2026.08.12]：面向鲲鹏系列处理器，发布针对社区版llama.cpp（commit `3ac67535c86`）的CPU算子优化版本，master 分支已合入 FP16 / FP32 / Q8_0 矩阵乘法与融合 SDPA 注意力算子的全部优化补丁，源码可直接编译运行。

## 项目介绍

llama-CPP是针对鲲鹏920B（7280Z）处理器进行的CPU小模型推理性能优化项目，聚焦llama.cpp推理引擎CPU后端的ARM矩阵乘法与注意力（SDPA）算子，采用ARM NEON / SVE-256 / i8mm指令集充分释放鲲鹏920B的算力，提升矩阵乘法和注意力计算性能。

本仓库 master 分支基于社区版llama.cpp（commit `3ac67535c86`）构建，已将 `opt` 分支中的全部 10 个优化补丁（`0001` ~ `0010`）合入源码，用户 clone master 后无需再手动 apply 补丁，可直接编译构建。优化能力主要包括：FP16 / FP32 / Q8_0三种数据类型的矩阵乘法kernel、Fused SDPA（FlashAttention v2 NEON融合算子）以及若干基线功能性修复。

## 目录结构

```text
llama-CPP/                                                                    # master 分支：合入优化后的源码
├── ggml                                                                      # ggml 算子库（含优化后的 CPU kernel）
│   ├── include/                                                              # ggml 头文件
│   ├── src/
│   │   ├── ggml.c                                                            # ggml 主实现
│   │   ├── ggml-cpu/                                                         # ggml CPU 后端（优化重点）
│   │   │   ├── ggml-cpu.c                                                    # 矩阵乘法 / 算子路由
│   │   │   ├── ggml-cpu-quants.c                                             # Q8_0 / MMLA 量化算子
│   │   │   ├── ops.cpp / ops.h                                               # fused SDPA 算子入口
│   │   │   ├── vec.cpp / vec.h                                               # ARM SVE / NEON FP16 / FP32 微内核
│   │   │   └── fused-cpp/fp32_packqkv/                                       # FP32 fused SDPA 前端
│   │   └── ...                                                               # 其它 ggml 后端
│   └── ...
├── src/                                                                       # llama.cpp 主实现
│   ├── llama-context.cpp                                                     # 推理上下文（合入调试 / 精度基线修复）
│   ├── llama-graph.cpp / llama-graph.h                                       # 计算图（接入 fused SDPA op）
│   ├── llama-model.cpp                                                       # 模型加载
│   └── ...
├── common/                                                                    # 公共库
│   ├── common.cpp / common.h                                                 # 公共接口（合入 embedding 修复）
│   └── CMakeLists.txt                                                        # 链接 OpenMP 等
├── tools/                                                                     # 可执行工具
│   ├── server/                                                               # llama-server（含 embedding 修复）
│   ├── llama-bench/
│   ├── llama-embedding/
│   └── ...
├── examples/                                                                  # 示例程序
├── tests/                                                                     # 单元测试
├── docs/                                                                      # 官方 llama.cpp 文档
│   ├── build.md
│   ├── development/
│   └── ...
├── compile.sh                                                                 # ARM 鲲鹏编译脚本（来自 0009 / 0010 补丁）
├── CMakeLists.txt                                                             # 顶层构建脚本
├── Makefile                                                                   # 顶层 Makefile
├── LICENSE                                                                    # 开源许可证文件（MIT）
├── README.md                                                                  # 项目说明文档
└── README_en.md                                                               # 英文项目说明文档
```

> 备注：master 分支的 `patch/` 目录请参见 `opt` 分支，该分支维护增量补丁。

## 版本说明

llama-CPP本身的版本说明，具体请参见《[版本说明书](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/zh/release_notes.md)》（`opt` 分支 docs/zh/release_notes.md）。

## 学习文档

| 资源名称 | 资源简介 |
| ------------ | ------------ |
| [版本说明书](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/zh/release_notes.md) | 提供llama-CPP每个发布版本的基础信息和特性更新信息。 |
| [特性介绍](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/zh/feature_introduction.md) | 提供llama-CPP优化说明。 |
| [用户指南](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/zh/user_guide.md) | 提供llama-CPP优化使用说明。 |
| [技术报告](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/zh/技术报告.md) | 提供精度与性能验证数据。 |
| [设计摘要](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/zh/设计摘要.md) | 提供各 kernel 的算法、接口与实现方案的摘要。 |
| [官方llama.cpp文档](./docs/) | 上游社区的构建、API 与开发文档。 |

## 快速开始

1. 拉取本仓库 master 分支源码：

   ```shell
   git clone https://gitcode.com/boostkit/llama-CPP.git
   cd llama-CPP
   ```

   master 分支已合入全部 10 个优化补丁（`0001` ~ `0010`），无需再手动 apply。

2. （可选）切到上游基线版本查看差异：基线 commit 为 `3ac67535c86`，对应优化补丁请见 `opt` 分支 `patch/` 目录。

3. 在鲲鹏920B（7280Z）服务器上编译：

   ```shell
   ./compile.sh build-delivery
   ```

   通用编译方式：

   ```shell
   _FLAGS="-O3 -funroll-loops"; env CFLAGS="$_FLAGS" CXXFLAGS="$_FLAGS" \
       cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build -GNinja \
             -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
   cmake --build build --config release -j 20
   ```

4. 启动 llama-server（embedding 示例）：

   ```shell
   taskset -c $(seq -s, 128 2 158) env OMP_NUM_THREADS=16 \
       ./build/bin/llama-server --model /path/to/bge-m3-FP16.gguf \
       --embedding --port 9180 --host 0.0.0.0 \
       --ctx-size 8192 --threads 16
   ```

5. 性能 / 精度验证脚本：

   - 整机性能：`bench_bgem3_full.py`（位于 `opt` 分支）
   - C-MTEB 精度：`eval_llamacpp_cmteb.py`（位于 `opt` 分支）

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交issues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 免责声明

此代码仓计划参与llama.cpp开源组件，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## 许可证书

本项目上游代码（llama.cpp）采用 MIT License，详见 [LICENSE](./LICENSE) 文件。
本项目文档适用CC-BY 4.0许可证。

## 致谢

感谢来自社区的每一个PR，欢迎贡献llama-CPP！
