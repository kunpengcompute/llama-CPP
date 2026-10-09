# llama.cpp-based BGE Model Tuning User Guide

## Tuning Overview

This document provides tuning guidance based on Kunpeng 950 processors/new Kunpeng 920 processor models. It describes how to apply optimization patches and compile and deploy the code based on the official llama.cpp repository (commit `3ac67535c86`), as well as how to run embedding models such as bge-m3.

## Environment Requirements

- Hardware platform: Kunpeng 950 processor/New Kunpeng 920 processor (NEON/SVE-256/i8mm).
- OS: openEuler 22.03 LTS SP3 or later.
- Compiler: GCC/G++ 12.3.1 or later (compatibility with `armv8.6-a+dotprod+i8mm+sve` required).
- Build dependencies: cmake >= 3.20, ninja (recommended), and git.
- Model selection: embedding models such as bge-m3 and bge-small-zh-v1.5 (FP16/Q8_0 GGUF format).

## Optimization Patches

The optimization patches are based on the official llama.cpp repository (commit `3ac67535c86`).

1. Pull the code. The following uses `/home/code` as an example.

   ```bash
   mkdir -p /home/code
   cd /home/code
   git clone <llama-CPP_repository_address> && cd llama-CPP
   ```

2. Apply the optimization patches.

   ```bash
   git am /home/code/llama-CPP/patch/00*.patch
   ```

## Optimization Patch Compilation

Use `compile.sh` for one-click compilation (`compile.sh` is introduced together with patches `0009` and `0010`).

```bash
./compile.sh build-delivery
```

> ![icon-note](./public_sys-resources/icon-note.gif)**NOTE:**
>By default, the compilation (`-O3 RelWithDebInfo`) builds ggml-cpu/llama-embedding/llama-bench/llama-server.

The compilation results are located in `build-delivery/bin/`.

## Service Launching

```bash
taskset -c $(seq -s, 128 2 158) env \
    GGML_FUSED_CPP_SDPA=1 \
    GGML_TOTAL_THREADS=16 OMP_NUM_THREADS=16 \
    ./build-delivery/bin/llama-server \
    --model /path/to/bge-m3-FP16.gguf \
    --embedding --port 9180 --host 0.0.0.0 \
    --ctx-size 8192 --threads 16 --pooling cls
```

The `GGML_FUSED_CPP_SDPA` environment variable controls whether to enable the fused SDPA path. It is enabled by default. Enabling it delivers better performance and is recommended.

| Environment Variable Value | Effect |
| ---------- | ---- |
| <code>0</code>/<code>off</code>/<code>false</code>/<code>no</code> | Disables fused SDPA and falls back to the official flash_attn. |
| <code>debug</code>/<code>trace</code> | Enables fused SDPA debug logging. |
| Empty/Other | Enabled by default. |

## Performance Verification

### Single-Core Performance

```bash
taskset -c 128 env \
    GGML_FUSED_CPP_SDPA=1 GGML_TOTAL_THREADS=1 OMP_NUM_THREADS=1 \
    ./build-delivery/bin/llama-bench -m /path/to/bge-m3-FP16.gguf -p 512 -b 512 -n 0 -t 1 -r 1
```

### Full-System Performance

After modifying the file path at the top of `bench_bgem3_full.py`, run it. The highest TPS is the optimal full-system performance.

```bash
vim bench_bgem3_full.py
python bench_bgem3_full.py
```

## Accuracy Verification

Before accuracy verification, prepare the official llama.cpp source code (commit `3ac67535c86`) and use `patch/baseline-bug-fixed.patch` to fix the baseline issues in it.

1. Obtain the official llama.cpp source code and check out the corresponding commit.

   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git /home/code/llama.cpp-baseline
   cd /home/code/llama.cpp-baseline
   git checkout 3ac67535c86
   ```

2. Apply the baseline fix patch and compile the code.

   ```bash
   git apply /home/code/llama-CPP/patch/baseline-bug-fixed.patch
   _FLAGS="-O3 -funroll-loops"
   env CFLAGS="$_FLAGS" CXXFLAGS="$_FLAGS" \
   cmake -DCMAKE_BUILD_TYPE=Release \
               -B build-baseline-fixed -GNinja \
               -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
   cmake --build build-baseline-fixed --target ggml-cpu llama-embedding llama-bench llama-server -j 20
   ```

3. After starting the server with this version, verify accuracy using the following commands.

```bash
python eval_llamacpp_cmteb.py \
--servers baseline=http://<ip>:<port> opt=<ip>:<port> \
--server-workers 2 \
--task-names   TNews IFlyTek MultilingualSentiment JDReview OnlineShopping Waimai     CLSClusteringS2S.v2 CLSClusteringP2P.v2 ThuNewsClusteringS2S.v2 ThuNewsClusteringP2P.v2     Ocnli Cmnli     T2Reranking MMarcoReranking CMedQAv1-reranking CMedQAv2-reranking     ATEC BQ LCQMC PAWSX STSB AFQMC QBQTC   \
--cmteb-root /path/to/C-MTEB \
--batch-size 1 \
--max-chars 500 \
--skip-bad-embedding \
--output-dir ./f16-all \
--continue-on-error
```

> ![icon-note](./public_sys-resources/icon-note.gif)**NOTE:**
>
>- Set the llama servers following `servers` in the format of `<server_name>=<server url>`. Multiple servers are separated by spaces.
>- The number of server-workers equals the number of servers above; use one worker for each server being tested.
>- Change the path following `cmteb-root` to your own C-MTEB dataset path.
>- If a task's score is `NaN`, you can run that task again.


## Change History

| Version | Date | Description |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | This is the first official release. |
