# 基于llama.cpp的BGE模型调优用户指南

## 调优概述

本文档基于鲲鹏950/鲲鹏920B平台，提供针对官方llama.cpp（commit `3ac67535c86`）合入优化补丁、编译、部署并运行 bge-m3 等 Embedding 模型的调优指导步骤。

## 环境要求

- 硬件平台：鲲鹏950/鲲鹏920B（NEON / SVE-256 / i8mm）
- 操作系统：openEuler 22.03 LTS SP3
- 编译器：GCC/G++ 15.2.0或更新版本（需支持 `armv8.6-a+dotprod+i8mm+sve`）
- 构建依赖：cmake >= 3.20、ninja（推荐）、git
- 模型选择：bge-m3、bge-small-zh-v1.5等Embedding模型（FP16/Q8_0 GGUF格式）

## 适配优化补丁

本优化补丁基于官方llama.cpp（commit `3ac67535c86`），与镜像`swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/llama.cpp:920B-kunpeng`中的commit 版本一致。

1. 拉取代码，此处以`/home/code`为例。

   ```bash
   mkdir -p /home/code
   cd /home/code
   git clone <llama-CPP 仓库地址> && cd llama-CPP
   ```

2. 合入优化补丁（先运行检查，再正式合入）。

   ```bash
   git apply --check /home/code/llama-CPP/patch/00*.patch    
   git am /home/code/llama-CPP/patch/00*.patch
   ```

   应用时会出现trailing whitespace警告，这是源码本身带的行尾空格（如 `server.cpp`、`quants.h`），不影响结果。

## 编译

使用`compile.sh`一键编译（`compile.sh` 由补丁`0009`、`0010`一并引入）。

```bash
./compile.sh build-delivery
```

>**说明：**
>默认编译（-O3 RelWithDebInfo，构建ggml-cpu/llama-embedding/llama-bench/ llama-server）

编译结果位于`build-delivery/bin/`。

## 启动服务

```bash
taskset -c $(seq -s, 128 2 158) env \
    GGML_FUSED_CPP_SDPA=1 \
    GGML_TOTAL_THREADS=16 OMP_NUM_THREADS=16 \
    ./build-delivery/bin/llama-server \
    --model /path/to/bge-m3-FP16.gguf \
    --embedding --port 9180 --host 0.0.0.0 \
    --ctx-size 8192 --threads 16 --pooling cls
```

`GGML_FUSED_CPP_SDPA`环境变量用于控制fused SDPA路径启停，默认开启。开启后性能更佳，推荐开启。

| 环境变量值 | 效果 |
| ---------- | ---- |
| `0` / `off` / `false` / `no` | 关闭fused SDPA，回退到官方flash_attn |
| `debug` / `trace` | 启用fused SDPA调试日志 |
| 空 /其他 | 默认启用 |

## 性能验证

### 单核性能

```bash
taskset -c 128 env \
    GGML_FUSED_CPP_SDPA=1 GGML_TOTAL_THREADS=1 OMP_NUM_THREADS=1 \
    ./build-delivery/bin/llama-bench -m /path/to/bge-m3-FP16.gguf -p 512 -b 512 -n 0 -t 1 -r 1
```

### 整机性能

修改`bench_bgem3_full.py`顶部的文件路径后运行，最高TPS即为整机最优性能。

```bash
vim bench_bgem3_full.py   
python bench_bgem3_full.py
```

## 精度验证

精度验证前，需先准备官方llama.cpp（commit `3ac67535c86`）源码，并使用`patch/baseline-bug-fixed.patch`修复其中的基线问题。

1. 获取官方llama.cpp源码并checkout到对应commit。

   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git /home/code/llama.cpp-baseline
   cd /home/code/llama.cpp-baseline
   git checkout 3ac67535c86
   ```

2. 应用基线修复补丁并编译。

   ```bash
   git apply /home/code/llama-CPP/patch/baseline-bug-fixed.patch
   _FLAGS="-O3 -funroll-loops"
   env CFLAGS="$_FLAGS" CXXFLAGS="$_FLAGS" \
   cmake -DCMAKE_BUILD_TYPE=Release \
               -B build-baseline-fixed -GNinja \
               -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
   cmake --build build-baseline-fixed --config release --target ggml-cpu llama-embedding llama-bench llama-server -j 20 
   ```

3. 用该版本启动server后，按下面的命令验证精度。

```bash
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

>**说明：**
>
>- servers后面以`server_name>=<server url`设置llama server，使用空格分离多个server。
>- server-workers数量等于上面的server数量，测试几个server就用几个worker。
>- cmteb-root改成自己的C-MTEB数据集路径。
>- 如果Task的分数是NaN，可以再重新运行该Task。

详细的kernel接口、算法与测试标准，请参见《[技术报告](./technical_report.md)》与《[特性介绍](./feature_introduction.md)》。

## 修订记录

| 发布日期 | 修订记录 |
| :--- | :--- |
| 2026-08-24 | 第一次正式发布。<br>- 统一llama.cpp相关描述为官方llama.cpp（commit `3ac67535c86`）。<br>- 处理器名称统一为鲲鹏950/鲲鹏920B，并移除处理器示例表述。 |
