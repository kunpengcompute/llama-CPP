# Release Notes

<!-- md-trans-meta sourceCommit=e87ebdab1b86243b748aef49f4a5f26cc32e1a67 translatedAt=2026-10-21T03:36:20.723Z pushedAt=2026-10-22T01:51:28.295Z nl=2 -->

## Version Mapping

### Product Version Information

|      |  |
| :--- | :--- |
| Product Name|Kunpeng BoostKit |  
|Version | 26.2.RC1 |
|Software Name|llama-CPP|
|Software Version| V1.0.0 |

### OS, Compiler, and CPU

| OS | Compiler | CPU Type | Baseline Version |
| :--- | :--- | :--- | :--- |
| openEuler 22.03 LTS SP3 | GCC/G++ 15.2.0 or later | Kunpeng 950 processor/New Kunpeng 920 processor model | llama.cpp (commit `3ac67535c86`) |

## V1.0.0

### Change Description

llama-CPP is a collection of CPU operator performance optimization patches for Kunpeng 950 processors/new Kunpeng 920 processor models. It focuses on the Arm matrix multiplication and attention operators of the llama.cpp inference engine's CPU backend. It uses the Arm NEON/SVE-256/i8mm instruction sets to implement Kunpeng-affinity optimizations, improving matrix multiplication and attention computation performance.

Key optimizations include:

- FP16 matrix multiplication optimization: NEON software-pipelined dot product, 4×4 outer-product tile, and outer-packA 8×16 hand-written outer-product kernels.
- FP32 matrix multiplication optimization: SVE 4×4 tile kernel.
- Q8_0 quantized matrix multiplication optimization: MMLA (i8mm) + `spack` data packing.
- Fused SDPA (FlashAttention v2 NEON fused operator).
- Fixes for the double-precision embedding normalize/similarity and the server output path issues in the official llama.cpp.

### Resolved Issues

Resolved the poor performance of FP16 and Q8_0 operators on Kunpeng 950 processors/new Kunpeng 920 processor models.

### Known Issues

None

## Related Documentation

### V1.0.0 Documentation

| Document | Description | Delivery Method |
| :--- | :--- | :--- |
| Release Notes | Provides basic information and feature update information for each release version of llama-CPP. | Open-source repository |
| Quick Start | Provides a quick guide for checking out code, applying the optimization patches, and compiling the code based on the official llama.cpp (commit <code>3ac67535c86</code>). | Open-source repository |
| User Guide | Provides usage instructions for llama-CPP. | Open-source repository |
| Feature Introduction | Provides an introduction to llama-CPP. | Open-source repository |
| Technical Report | Provides accuracy and performance verification data. | Open-source repository |

### Obtaining Documentation

Visit the [open-source repository](https://gitcode.com/boostkit/llama-CPP/blob/opt/docs) to view or download related documents.

## Change History

| Version | Date | Description |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | This is the first official release. |
