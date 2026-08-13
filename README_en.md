# Introduction to llama-CPP

## Latest Updates

- [2026-08-12]: Released a collection of CPU operator optimization patches for the community llama.cpp (commit `3ac67535c86`), targeting Kunpeng processors, focusing on FP16 / FP32 / Q8_0 matrix multiplication and fused SDPA attention operators.

## Project Introduction

llama-CPP is a performance improvement project for small-model CPU inference based on Kunpeng processors. It focuses on the ARM matrix multiplication and attention (SDPA) operators in the llama.cpp inference engine CPU backend, leveraging ARM NEON / SVE-256 / i8mm instructions to fully unleash the computing power of Kunpeng processors and improve matrix multiplication and attention performance. This project provides optimization patches for the community llama.cpp.

Main capabilities include: FP16 / FP32 / Q8_0 matrix multiplication kernels, Fused SDPA (FlashAttention v2 NEON fused operator), and several baseline functional fixes.

## Directory Structure

```text
llama-CPP/
├── patch                                                                    # Patch file directory
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
│   └── baseline-bug-fixed.patch                                            # Baseline fix patch for accuracy verification
├── docs
│   ├── zh                                                                    # Chinese document directory
│   │   ├── feature_introduction.md                                            # Feature description document
│   │   ├── menu_llamacpp.md                                                   # Document guide
│   │   ├── release_notes.md                                                   # Release notes
│   │   ├── user_guide.md                                                      # User guide
│   │   ├── unit_test_guide.md                                                 # Unit test guide
│   │   ├── technical_report.md                                                        # Accuracy and performance verification data
│   └── en                                                                    # English document directory
│       ├── feature_introduction.md
│       ├── menu_llamacpp.md
│       ├── release_notes.md
│       ├── user_guide.md
│       └── unit_test_guide.md
├── bench_bgem3_full.py                                                      # Whole-machine performance verification script
├── eval_llamacpp_cmteb.py                                                   # C-MTEB accuracy verification script
├── LICENSE                                                                   # Open-source license file (Apache 2.0)
├── CC-BY                                                                     # Open-source document license file (CC-BY 4.0)
├── README.md                                                                 # Project introduction
└── README_en.md                                                              # English project introduction
```

## Release Notes

For details about the llama-CPP version description, see [Release Notes](./docs/en/release_notes.md).

## Documents

| Document Name | Description |
| ------------ | ------------ |
| [Release Notes](./docs/en/release_notes.md) | Provides basic information and feature updates of each llama-CPP version. |
| [Feature Introduction](./docs/en/feature_introduction.md) | Provides llama-CPP optimization description. |
| [User Guide](./docs/en/user_guide.md) | Provides llama-CPP optimization usage description. |
| [Unit Test Guide](./docs/en/unit_test_guide.md) | Provides instructions for building, running, and expected output of the operator optimization unit tests. |

The Chinese technical report (with accuracy and performance data) is also available under `docs/zh/`.

## Quick Start

1. Check out the official llama.cpp commit `3ac67535c86`;
2. Apply the patches: `git am patch/00*.patch`;
3. Build: `./compile.sh build-delivery`;
4. For performance/accuracy verification, see the [User Guide](./docs/en/user_guide.md).

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit issues. For details, see the [contribution guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Disclaimer

This code repository contributes to the llama.cpp open-source component. It strictly adheres to the coding style and methods, as well as security design, of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

This project is released under the Apache License 2.0. For details, see [LICENSE](./LICENSE).
The documents of this project are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

## Acknowledgments

llama-CPP is jointly developed by the following Huawei department:

Kunpeng Computing BoostKit Development Dept

Thank you to everyone in the community for your PRs. We warmly welcome contributions to llama-CPP!
