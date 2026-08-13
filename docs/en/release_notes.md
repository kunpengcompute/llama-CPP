# Release Notes

## Version Mapping

### Product Version Information

| Product Name | Product Version | Software Name | Software Package Version |
| :--- | :--- | :--- | :--- |
| Kunpeng BoostKit | 26.8.RC1 | llama-CPP | V1.0.0 |

### OS, Compiler, and CPU

| Operating System | Compiler | CPU Type | Baseline Version |
| :--- | :--- | :--- | :--- |
| openEuler 22.03 LTS SP3 | GCC/G++ 15.2.0 or later | Kunpeng 920B (7280Z) | llama.cpp commit `3ac67535c86` |

## V1.0.0

### Change Description

llama-CPP is a collection of CPU operator performance optimization patches for the Kunpeng 920B (7280Z) processor. It focuses on the ARM matrix multiplication and attention operators in the llama.cpp inference engine CPU backend, using ARM NEON / SVE-256 / i8mm instructions to achieve Kunpeng-affinity optimization and improve matrix multiplication and attention performance.

Key capabilities:

- FP16 matrix multiplication optimization: NEON software-pipelined dot product, 4×4 outer-product tile, and outer-packA 8×16 hand-written outer-product kernels;
- FP32 matrix multiplication optimization: SVE 4×4 tile kernel;
- Q8_0 quantized matrix multiplication optimization: MMLA (i8mm) + Spack data packing;
- Fused SDPA (FlashAttention v2 NEON fused operator) and diagonal-block optimization;
- Baseline fixes: double-precision embedding normalize / similarity accumulation and server output-path fixes.

### Resolved Issues

None

### Known Issues

None

## Related Documentation

### V1.0.0 Documentation

| Document | Description | Delivery Mode |
| :--- | :--- | :--- |
| Release Notes | Provides basic information and feature updates of each llama-CPP version. | Open-source repository |
| User Guide | Provides llama-CPP optimization usage description. | Open-source repository |
| Feature Introduction | Provides llama-CPP optimization description. | Open-source repository |
| Technical Report | Provides accuracy and performance verification data. | Open-source repository |
| Design Summary | Provides a summary of the algorithm and interfaces of each kernel. | Open-source repository |

### Obtaining Documentation

Visit the llama-CPP open-source repository to view or download related documents.
