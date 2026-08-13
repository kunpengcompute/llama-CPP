# User Guide for BGE Model Performance Optimization Based on llama.cpp

This document provides tuning guidance for applying the optimization patches, building, deploying, and running Embedding models such as bge-m3 on Kunpeng processors (using the Kunpeng 920B as an example) with the community llama.cpp (commit `3ac67535c86`).

## Environment Requirements

- Hardware platform: Kunpeng processor (e.g., Kunpeng 920B; NEON / SVE-256 / i8mm)
- Operating system: openEuler 22.03 LTS SP3
- Compiler: GCC/G++ 15.2.0 or later (must support `armv8.6-a+dotprod+i8mm+sve`)
- Build dependencies: cmake >= 3.20, ninja (recommended), git
- Model selection: Embedding models such as bge-m3 and bge-small-zh-v1.5 (FP16 / Q8_0 GGUF format)

## Optimization Patches

The patches are based on the official llama.cpp commit `3ac67535c86`, matching the commit version in the image `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/llama.cpp:920B-kunpeng`.

1. Pull the code. The following uses `/home/code` as an example.

   ```bash
   mkdir -p /home/code
   cd /home/code
   git clone <llama-CPP repository URL> && cd llama-CPP
   ```

2. Apply the optimization patches (optionally dry-run first, then apply officially).

   ```bash
   git apply --check /home/code/llama-CPP/patch/00*.patch    # optional: dry-run check
   git am /home/code/llama-CPP/patch/00*.patch
   ```

   A few trailing-whitespace warnings may appear during application; these come from whitespace at line ends in the source itself (for example, `server.cpp` and `quants.h`) and do not affect the result.

## Build

Use `compile.sh` for one-click building (`compile.sh` is introduced together with patches `0009` and `0010`).

```bash
# Default build (-O3 RelWithDebInfo; builds ggml-cpu / llama-embedding / llama-bench / llama-server)
./compile.sh build-delivery
```

Build outputs are located in `build-delivery/bin/`.

## Start the Service

```bash
taskset -c $(seq -s, 128 2 158) env \
    GGML_FUSED_CPP_SDPA=1 \
    GGML_TOTAL_THREADS=16 OMP_NUM_THREADS=16 \
    ./build-delivery/bin/llama-server \
    --model /path/to/bge-m3-FP16.gguf \
    --embedding --port 9180 --host 0.0.0.0 \
    --ctx-size 8192 --threads 16 --pooling cls
```

The `GGML_FUSED_CPP_SDPA` environment variable controls the fused SDPA path. It is enabled by default and recommended for better performance:

| Environment Variable Value | Effect |
| -------------------------- | ------ |
| `0` / `off` / `false` / `no` | Disable fused SDPA and fall back to native flash_attn |
| `debug` / `trace` | Enable fused SDPA debug logs |
| Empty / other | Enabled by default |

## Performance Verification

### Single Core

```bash
taskset -c 128 env \
    GGML_FUSED_CPP_SDPA=1 GGML_TOTAL_THREADS=1 OMP_NUM_THREADS=1 \
    ./build-delivery/bin/llama-bench -m /path/to/bge-m3-FP16.gguf -p 512 -b 512 -n 0 -t 1 -r 1
```

### Whole Machine

Modify the file paths at the top of `bench_bgem3_full.py` and run it. The highest TPS is the optimal whole-machine performance.

```bash
vim bench_bgem3_full.py   # modify the file paths at the top
python bench_bgem3_full.py
```

## Accuracy Verification

Before accuracy verification, prepare the official baseline source and fix the baseline using `patch/baseline-bug-fixed.patch`.

1. Get the official llama.cpp source and check out the matching commit:

   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git /home/code/llama.cpp-baseline
   cd /home/code/llama.cpp-baseline
   git checkout 3ac67535c86
   ```

2. Apply the baseline fix patch and build:

   ```bash
   git apply /home/code/llama-CPP/patch/baseline-bug-fixed.patch
   _FLAGS="-O3 -funroll-loops"
   env CFLAGS="$_FLAGS" CXXFLAGS="$_FLAGS" \
   cmake -DCMAKE_BUILD_TYPE=Release \
               -B build-baseline-fixed -GNinja \
               -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
   cmake --build build-baseline-fixed --config release --target ggml-cpu llama-embedding llama-bench llama-server -j 20
   ```

3. Start the server with this version, then verify accuracy using the commands below.

```bash
# --servers is followed by <server_name>=<server url>, separated by spaces
# --server-workers equals the number of servers above (one worker per server)
# --cmteb-root is your C-MTEB dataset path
# If a task score is NaN, run that task separately

python eval_llamacpp_cmteb.py \
--servers baseline=http://141.61.21.62:8080 opt=http://141.61.21.62:7080 \
--server-workers 2 \
--task-names   TNews IFlyTek MultilingualSentiment JDReview OnlineShopping Waimai     CLSClusteringS2S.v2 CLSClusteringP2P.v2 ThuNewsClusteringS2S.v2 ThuNewsClusteringP2P.v2     Ocnli Cmnli     T2Reranking MMarcoReranking CMedQAv1-reranking CMedQAv2-reranking     ATEC BQ LCQMC PAWSX STSB AFQMC QBQTC   \
--cmteb-root /home/l30061571/models/datasets/C-MTEB \
--batch-size 1 \
--max-chars 500 \
--skip-bad-embedding \
--output-dir ./f16-all \
--continue-on-error
```

For detailed kernel interfaces, algorithms, and test criteria, see the technical report (`technical_report.md`) and feature introduction (`feature_introduction.md`) under `docs/zh/`.
