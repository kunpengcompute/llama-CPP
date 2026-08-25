# llama-CPP介绍

简体中文 | [English](./README_en.md)

## 最新消息

- [2026.09.30]：面向鲲鹏950/鲲鹏920B，发布针对官方llama.cpp（commit `3ac67535c86`）的CPU算子优化补丁合集，聚焦FP16 / FP32 / Q8_0矩阵乘法与融合SDPA注意力算子。

## 项目介绍

llama-CPP是针对鲲鹏950/鲲鹏920B进行的CPU小模型推理性能优化，聚焦llama.cpp推理引擎CPU后端的ARM矩阵乘法与注意力（SDPA）算子，采用ARM NEON/SVE-256/i8mm指令集充分释放鲲鹏950/鲲鹏920B的算力，提升矩阵乘法和注意力计算性能。本项目针对官方llama.cpp输出优化补丁。

优化能力主要包括：FP16/FP32/Q8_0三种数据类型的矩阵乘法kernel、Fused SDPA（FlashAttention v2 NEON融合算子）以及针对官方llama.cpp的若干功能性修复。

## 目录结构

```text
llama-CPP/
├── patch                                         # 补丁文件目录
│   ├── 0001-fix-common-fix-baseline-embedding-regex-serv-bugs-ad.patch
│   ├── 0002-feat-cpu-add-ARM-SVE-NEON-FP16-FP32-matmul-kernels.patch
│   ├── 0003-feat-cpu-add-Q8_0-MMLA-spack-matmul-kernels.patch
│   ├── 0004-feat-cpu-add-fused-SDPA-microkernels-neon-cache-QK-p.patch
│   ├── 0005-feat-cpu-add-fused-SDPA-flash2-L3KV-implementation.patch
│   ├── 0006-feat-cpu-add-fused-SDPA-fp32-packqkv-front-end.patch
│   ├── 0007-feat-llama-add-GGML-fused-SDPA-op-and-integrate-into.patch
│   ├── 0008-feat-cpu-wire-optimized-matmul-fused-SDPA-kernels-in.patch
│   ├── 0009-build-add-compile.sh-helper-for-ARM-builds.patch
│   ├── 0010-update-compile.sh.patch
│   ├── 0011-add-f16-q80-unit-test.patch
│   └── baseline-bug-fixed.patch                  # 精度验证基线修复补丁
├── docs
│   ├── zh                                        # 中文文档目录
│   │   ├── menu_llamacpp.md                      # 文档导航
│   │   ├── quick_start.md                        # 快速入门
│   │   ├── feature_introduction.md               # 特性指南
│   │   ├── release_notes.md                      # 版本说明书
│   │   ├── user_guide.md                         # 用户指南
│   │   ├── unit_test_guide.md                    # 单元测试指南
│   │   ├── technical_report.md                   # 技术报告
│   └── en                                        # English document directory
│       ├── feature_introduction.md               # feature_introduction
│       ├── release_notes.md                      # release_notes
│       ├── user_guide.md                         # user_guide
│       └── unit_test_guide.md                    # unit_test_guide
├── bench_bgem3_full.py                           # 整机性能验证脚本
├── eval_llamacpp_cmteb.py                        # C-MTEB精度验证脚本
├── LICENSE                                       # 开源许可证文件(Apache 2.0)
├── CC-BY                                         # 开源文档许可证文件(CC-BY 4.0)
├── README.md                                     # 项目说明
└── README_en.md                                  # 英文项目说明
```

## 版本说明

llama-CPP版本说明，请参见《[版本说明书](./docs/zh/release_notes.md)》。

## 学习文档

| 资源名称 | 资源简介 |
| ------------ | ------------ |
| [版本说明书](./docs/zh/release_notes.md) | 提供llama-CPP每个发布版本的基础信息和特性更新信息。 |
| [特性介绍](./docs/zh/feature_introduction.md) | 提供llama-CPP优化说明。 |
| [用户指南](./docs/zh/user_guide.md) | 提供llama-CPP优化使用说明。 |
| [单元测试指南](./docs/zh/unit_test_guide.md) | 提供算子优化单元测试的编译、执行与预期输出说明。 |
| [技术报告](./docs/zh/technical_report.md) | 提供精度与性能验证数据。 |
| [快速入门](./docs/zh/quick_start.md) | 提供基于官方llama.cpp（commit `3ac67535c86`）checkout代码、合入优化补丁与编译的快速指南。 |

## 快速入门

1. 获取官方llama.cpp源码，并checkout到commit `3ac67535c86`。
2. 在llama.cpp源码目录中合入优化补丁：`git am patch/00*.patch`。
3. 执行`./compile.sh build-delivery`完成编译。

详细的环境要求、操作步骤与验证方法，请参见《[快速入门](./docs/zh/quick_start.md)》。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交issues联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 免责声明

此代码仓计划参与llama.cpp开源组件，编码风格遵照开源软件，继承开源软件安全设计，不破坏开源软件设计及编码风格和方式。软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## 许可证书

本项目采用Apache License 2.0，详见[LICENSE](./LICENSE)文件。
本项目文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

## 致谢

感谢来自社区的每一个PR，欢迎贡献llama-CPP！
