# Quick Start

This document describes how to check out the code, apply the optimization patches, and perform compilation based on the official llama.cpp repository (commit `3ac67535c86`), so that users can quickly experience the CPU operator optimization capabilities provided by llama-CPP on Kunpeng 950 processors/new Kunpeng 920 processor models. The optimization patches focus on FP16/FP32/Q8_0 matrix multiplication and fused Scaled Dot-Product Attention (SDPA) operators, which can improve the performance of llama.cpp in CPU-based small-model inference.

## Environment Requirements

- Hardware platform: Kunpeng 950 processor/New Kunpeng 920 processor (NEON/SVE-256/i8mm).
- OS: openEuler 22.03 LTS SP3 or later.
- Compiler: GCC/G++ 12.3.1 or later (compatibility with `armv8.6-a+dotprod+i8mm+sve` required).
- Build dependencies: cmake >= 3.20, ninja (recommended), and git.

## Code Pulling and Commit Checkout

The optimization patches are based on the official llama.cpp repository (commit `3ac67535c86`). First switch the source code to this commit and pin it.

1. Pull the official llama.cpp source code.

   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   ```

2. Check out the version corresponding to the official llama.cpp repository (commit `3ac67535c86`).

   ```bash
   git checkout 3ac67535c86
   ```

## Optimization Patch Integration

llama-CPP patch files `0001` to `0011` (`00*.patch`) are located in the `patch/` directory of the repository. Apply them in the llama.cpp source code directory.

1. Obtain the llama-CPP repository (if not already obtained).

   ```bash
   git clone <llama-CPP_repository_address>
   cd <llama-CPP_repository_path>
   ```

2. Apply the optimization patches.

   ```bash
   git am /path/to/llama-CPP/patch/00*.patch
   ```

## Compilation

Use `compile.sh` for one-click compilation.

```bash
./compile.sh build-delivery
```

>![icon-note](./public_sys-resources/icon-note.gif)**NOTE:**
>By default, the compilation (`-O3 RelWithDebInfo`) builds ggml-cpu, llama-embedding, llama-bench, and llama-server. The compilation results are located in `build-delivery/bin/`.

## Verification and Further Reading

After compilation is complete, you can run unit tests to verify the correctness of the optimized operators.

```bash
./build-delivery/bin/test-sdpa-f16q80-opt
```

## Learning Documents

- [User Guide](./user_guide.md): Deployment and tuning of embedding models such as bge-m3.
- [Feature Introduction](./feature_introduction.md): Features and principles of the optimized operators.
- [Unit Test Guide](./unit_test_guide.md): Compilation, execution, and expected output of unit tests.
- [Release Notes](./release_notes.md): Release version information and related documents.

## Change History

| Version | Date | Description |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | This is the first official release. |
