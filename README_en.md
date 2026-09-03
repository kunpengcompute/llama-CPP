# Introduction to llama-CPP

## Latest Updates

- [2026-08-12]: Released the CPU operator optimization version of community llama.cpp (commit `3ac67535c86`) targeting the Kunpeng 920B (7280Z) processor. The `master` branch already incorporates all 10 optimization patches (`0001`–`0010`) covering FP16 / FP32 / Q8_0 matrix multiplication and fused SDPA attention operators, and the source can be built directly.

## Project Introduction

llama-CPP is a performance improvement project for small-model CPU inference based on the Kunpeng 920B (7280Z) processor. It focuses on the ARM matrix multiplication and attention (SDPA) operators in the llama.cpp inference engine CPU backend, leveraging ARM NEON / SVE-256 / i8mm instructions to fully unleash the computing power of the Kunpeng 920B and improve matrix multiplication and attention performance.

The `master` branch of this repository is built on top of community llama.cpp (commit `3ac67535c86`) with all 10 optimization patches from the `opt` branch (`0001`–`0010`) already merged into the source tree. Users can clone `master` and build directly without applying any patches manually. Main capabilities include: FP16 / FP32 / Q8_0 matrix multiplication kernels, Fused SDPA (FlashAttention v2 NEON fused operator), and several baseline functional fixes.

## Directory Structure

```text
llama-CPP/                                                                    # master branch: optimized source tree
├── ggml                                                                      # ggml operator library (with optimized CPU kernels)
│   ├── include/                                                              # ggml headers
│   ├── src/
│   │   ├── ggml.c                                                            # ggml core
│   │   ├── ggml-cpu/                                                         # ggml CPU backend (focus of the optimization)
│   │   │   ├── ggml-cpu.c                                                    # mat-mul / op routing
│   │   │   ├── ggml-cpu-quants.c                                             # Q8_0 / MMLA quantized operators
│   │   │   ├── ops.cpp / ops.h                                               # fused SDPA op entry
│   │   │   ├── vec.cpp / vec.h                                               # ARM SVE / NEON FP16 / FP32 microkernels
│   │   │   └── fused-cpp/fp32_packqkv/                                       # FP32 fused SDPA front-end
│   │   └── ...                                                               # other ggml backends
│   └── ...
├── src/                                                                       # llama.cpp core
│   ├── llama-context.cpp                                                     # inference context (with debug / baseline fixes)
│   ├── llama-graph.cpp / llama-graph.h                                       # compute graph (wires fused SDPA op)
│   ├── llama-model.cpp                                                       # model loader
│   └── ...
├── common/                                                                    # common library
│   ├── common.cpp / common.h                                                 # common APIs (with embedding fixes)
│   └── CMakeLists.txt                                                        # links OpenMP, etc.
├── tools/                                                                     # executables
│   ├── server/                                                               # llama-server (with embedding fixes)
│   ├── llama-bench/
│   ├── llama-embedding/
│   └── ...
├── examples/                                                                  # example programs
├── tests/                                                                     # unit tests
├── docs/                                                                      # official llama.cpp documentation
│   ├── build.md
│   ├── development/
│   └── ...
├── compile.sh                                                                 # ARM Kunpeng build script (from patches 0009 / 0010)
├── CMakeLists.txt                                                             # top-level build script
├── Makefile                                                                   # top-level Makefile
├── LICENSE                                                                    # upstream license (MIT)
├── README.md                                                                  # project introduction
└── README_en.md                                                               # English project introduction
```

> Note: the `patch/` directory is maintained on the `opt` branch; please refer to it for the upstream-style incremental patch set.

## Release Notes

For details about the llama-CPP version description, see [Release Notes](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/en/release_notes.md) on the `opt` branch.

## Documents

| Document Name | Description |
| ------------ | ------------ |
| [Release Notes](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/en/release_notes.md) | Provides basic information and feature updates of each llama-CPP version. |
| [Feature Introduction](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/en/feature_introduction.md) | Provides llama-CPP optimization description. |
| [User Guide](https://gitcode.com/wangyan575757/llama-CPP/blob/opt/docs/en/user_guide.md) | Provides llama-CPP optimization usage description. |
| [Official llama.cpp docs](./docs/) | Upstream build, API and development documentation. |

The Chinese technical report and detailed design document are also available under `opt/docs/zh/`.

## Quick Start

1. Clone the `master` branch of this repository:

   ```shell
   git clone https://gitcode.com/boostkit/llama-CPP.git
   cd llama-CPP
   ```

   The `master` branch already merges all 10 optimization patches (`0001`–`0010`); no manual patch application is needed.

2. (Optional) Compare against the upstream baseline: the baseline commit is `3ac67535c86`. The corresponding optimization patches are kept under `opt/patch/`.

3. Build on a Kunpeng 920B (7280Z) server:

   ```shell
   ./compile.sh build-delivery
   ```

   Generic build:

   ```shell
   _FLAGS="-O3 -funroll-loops"; env CFLAGS="$_FLAGS" CXXFLAGS="$_FLAGS" \
       cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build -GNinja \
             -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
   cmake --build build --config release -j 20
   ```

4. Launch llama-server (embedding example):

   ```shell
   taskset -c $(seq -s, 128 2 158) env OMP_NUM_THREADS=16 \
       ./build/bin/llama-server --model /path/to/bge-m3-FP16.gguf \
       --embedding --port 9180 --host 0.0.0.0 \
       --ctx-size 8192 --threads 16
   ```

5. Performance / accuracy verification scripts:

   - End-to-end performance: `bench_bgem3_full.py` (on the `opt` branch)
   - C-MTEB accuracy: `eval_llamacpp_cmteb.py` (on the `opt` branch)

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit issues. For details, see the [contribution guideline](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md). You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Disclaimer

This code repository contributes to the llama.cpp open-source component. It strictly adheres to the coding style and methods, as well as security design, of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

The upstream llama.cpp code is released under the MIT License; see [LICENSE](./LICENSE).
The documents of this project are licensed under CC-BY 4.0.

## Acknowledgments

Thank you to everyone in the community for your PRs. We warmly welcome contributions to llama-CPP!
