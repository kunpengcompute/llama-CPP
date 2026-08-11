# llama-CPP

patch基于官方llama.cpp的commit 3ac67535c86，与`swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/llama.cpp:920B-kunpeng`镜像中的commit版本一致。

### patch合入

- patch合入：

```shell
git apply --check /path/to/patches/00*.patch    # 可选：先干跑检查
git am /path/to/patches/00*.patch
```
- 应用时会有几条 trailing whitespace 警告，那是源码本身带的行尾空格（比如 server.cpp、quants.h），不影响结果。

- 编译：

```shell
./compile.sh build-delivery
```

- 启动server（如需）：
```shell
taskset -c $(seq -s, 128 2 158) env GGML_FUSED_CPP_SDPA=1 GGML_TOTAL_THREADS=16 OMP_NUM_THREADS=16 ./build-delivery/bin/llama-server --model /path/to/bge-m3-FP16.gguf --embedding --port 9180 --host 0.0.0.0 --ctx-size 8192 --threads 16 --pooling cls
```


patch文件改动

```
  |-- common/
  |   |-- CMakeLists.txt  [mod]
  |   `-- common.cpp  [mod]
  |-- compile.sh  [+new]
  |-- ggml/
  |   |-- include/
  |   |   |-- ggml-cpu.h  [mod]
  |   |   `-- ggml.h  [mod]
  |   `-- src/
  |       |-- ggml.c  [mod]
  |       `-- ggml-cpu/
  |           |-- CMakeLists.txt  [mod]
  |           |-- ggml-cpu.c  [mod]
  |           |-- ggml-cpu-quants.c  [mod]
  |           |-- ggml-cpu-quants.h  [mod]
  |           |-- ops.cpp  [mod]
  |           |-- ops.h  [mod]
  |           |-- vec.cpp  [mod]
  |           |-- vec.h  [mod]
  |           `-- fused-cpp/
  |               `-- fp32_packqkv/
  |                   |-- fp32_packqkv_sdpa.cpp  [+new]
  |                   |-- fp32_packqkv_sdpa.h  [+new]
  |                   |-- parse_embedding_perf_log.py  [+new]
  |                   `-- csrc/
  |                       |-- sdpa_common.h  [+new]
  |                       |-- sdpa_flash2_neon_l3kv_impl.h  [+new]
  |                       |-- sdpa_pack_utils.h  [+new]
  |                       |-- sdpa_profile.h  [+new]
  |                       |-- sdpa_standalone_shim.h  [+new]
  |                       |-- sdpa_tile_sizes.h  [+new]
  |                       `-- sdpa_microkernels/
  |                           |-- mk_traits.h  [+new]
  |                           |-- neon_cache_config.h  [+new]
  |                           |-- neon_cache_microkernels.h  [+new]
  |                           `-- impls/
  |                               `-- mk_qk_packqk_seq4_bmajor_pv_pquad.h  [+new]
  |-- src/
  |   |-- llama-context.cpp  [mod]
  |   |-- llama-graph.cpp  [mod]
  |   |-- llama-graph.h  [mod]
  |   `-- llama-model.cpp  [mod]
  `-- tools/
      `-- server/
          |-- server.cpp  [mod]
          `-- utils.hpp  [mod]
```

### 快速验证

#### 性能
```shell
# 单核
taskset -c 128 env  GGML_FUSED_CPP_SDPA=1 GGML_TOTAL_THREADS=1 OMP_NUM_THREADS=1 ./build-delivery/bin/llama-bench -m /path/to/bge-m3-FP16.gguf -p 512 -b 512 -n 0 -t 1 -r 1
# 整机
vim bench_bgem3_full.py   #修改顶部的文件路径
python bench_bgem3_full.py   #最高TPS就是整机最优性能
```

#### 精度

 精度验证前务必将基线使用`baseline-buf-fixed.patch`修复。
 
 步骤：
 1. checkout到官方llama.cpp的commit 3ac67535c86
 2. git apply baseline-buf-fixed.patch
 3. 编译，用这个版本来验证精度
 
 
 ```shell
 # --servers后面以<server_name>=<server url>设置llama server，空格分离多个server
 # --server-workers数量等于上面的server数，测试几个server就用几个worker
 # --cmteb-root改成你的C-MTEB数据集路径
 # 有哪个Task的分数是NaN，可以再单独跑一下该Task
 
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
 