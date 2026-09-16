# llama-CPP Introduction

English|[简体中文](./README.md)

## Latest Updates

[2026-09-30]: Released a collection of CPU operator optimization patches targeting Kunpeng 950 processors/new Kunpeng 920 processor models for the official llama.cpp repository (commit `3ac67535c86`). The optimization focuses on FP16/FP32/Q8_0 matrix multiplication and fused Scaled Dot-Product Attention (SDPA) operators.

## Project Introduction

llama-CPP is a CPU-based small-model inference performance optimization for Kunpeng 950 processors/new Kunpeng 920 processor models. It uses the Arm NEON/SVE-256/i8mm instruction sets to fully unleash the computing power of Kunpeng 950 processors/new Kunpeng 920 processor models, thereby improving matrix multiplication and SDPA computation performance in the CPU backend of the llama.cpp inference engine. This project outputs optimization patches for the official llama.cpp repository.

The optimization capabilities mainly include: matrix multiplication kernels for three data types (FP16/FP32/Q8_0), fused SDPA (FlashAttention v2 NEON fused operator), and several functional fixes for the official llama.cpp repository.

## Directory Structure

```text
llama-CPP/
├── patch                                         # Patch file directory
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
│   └── baseline-bug-fixed.patch                  # Baseline fix patch for accuracy verification
├── docs
│   ├── zh                                        # Chinese documentation directory
│   │   ├── menu_llamacpp.md                      # Documentation navigation
│   │   ├── quick_start.md                        # Quick Start
│   │   ├── feature_introduction.md               # Feature Guide
│   │   ├── release_notes.md                      # Release Notes
│   │   ├── user_guide.md                         # User Guide
│   │   ├── unit_test_guide.md                    # Unit Test Guide
│   │   ├── technical_report.md                   # Technical Report
│   └── en                                        # English documentation directory
│       ├── menu_llamacpp.md                      # Documentation navigation
│       ├── feature_introduction.md               # Feature Guide
│       ├── release_notes.md                      # Release Notes
│       ├── user_guide.md                         # User Guide
│       └── unit_test_guide.md                    # Unit Test Guide
├── bench_bgem3_full.py                           # Full-system performance verification script
├── eval_llamacpp_cmteb.py                        # C-MTEB accuracy verification script
├── LICENSE                                       # Open-source license file (Apache 2.0)
├── CC-BY                                         # Open-source documentation license file (CC-BY 4.0)
├── README.md                                     # Project description (Chinese)
└── README_en.md                                  # Project description (English)
```

## Release Notes

For details about the llama-CPP release notes, see [Release Notes](./docs/en/release_notes.md).

## Related Documents

| Document | Description |
| ------------ | ------------ |
| [Release Notes](./docs/en/release_notes.md) | Provides basic information and feature update information for each llama-CPP release. |
| [Feature Introduction](./docs/en/feature_introduction.md) | Provides the llama-CPP optimization description. |
| [User Guide](./docs/en/user_guide.md) | Provides instructions on using the llama-CPP optimizations. |
| [Unit Test Guide](./docs/en/unit_test_guide.md) | Describes compilation, execution, and expected output of operator optimization unit tests. |
| [Technical Report](./docs/en/technical_report.md) | Provides accuracy and performance verification data. |
| [Quick Start](./docs/en/quick_start.md) | Provides a quick guide to checking out the official llama.cpp code (commit <code>3ac67535c86</code>), applying the optimization patches, and performing compilation. |

## Quick Start

1. Obtain the official llama.cpp source code and check out commit `3ac67535c86`.
2. Apply the optimization patches in the llama.cpp source code directory: `git am patch/00*.patch`.
3. Run `./compile.sh build-delivery` to complete the compilation.

For detailed environment requirements, operation steps, and verification methods, see [Quick Start](./docs/en/quick_start.md).

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit issues. For details, see the [contribution guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in the [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Disclaimer

This code repository contributes to the llama.cpp open-source component. It strictly adheres to the coding style and methods, as well as security design of the open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

This project is released under the Apache License 2.0. For details, see [LICENSE](./LICENSE).
The documents of this project are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

## Acknowledgments

Thank you to everyone in the community for your PRs. We warmly welcome contributions to llama-CPP!
