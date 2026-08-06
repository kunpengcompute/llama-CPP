# llama-CPP

`gemm_opt_for_fp16_fp32.patch`基于官方llama.cpp的commit 3ac67535c86，与`swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/llama.cpp:920B-kunpeng`镜像中的commit版本一致。

- patch合入：

```shell
git apply gemm_opt_for_fp16_fp32.patch
git add ggml/
git commit -m ""
```

- 编译：

```shell
_FLAGS="-O3 -funroll-loops"; env CFLAGS="$_FLAGS" CXXFLAGS="$_FLAGS" cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build -GNinja -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
cmake --build build --config release -j 20
```

- 启动server：
```shell
taskset -c $(seq -s, 128 2 158) env OMP_NUM_THREADS=16 ./build/bin/llama-server --model /path/to/bge-m3-FP16.gguf --embedding --port 9180 --host 0.0.0.0 --ctx-size 8192 --threads 16
```

- 性能验证
1. 快速验证：
```shell
taskset -c $(seq -s, 128 2 158) env OMP_NUM_THREADS=16 ./build/bin/llama-bench -m /path/to/bge-m3-FP16.gguf -p 256 -b 256 -n 0 -t 16 -r 50
```


- 文件改动
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
  |-- pocs/
  |   `-- vdot/
  |       |-- CMakeLists.txt  [mod]
  |       `-- test_q8_0_matmul.cpp  [+new]
  |-- src/
  |   |-- llama-context.cpp  [mod]
  |   |-- llama-graph.cpp  [mod]
  |   |-- llama-graph.h  [mod]
  |   `-- llama-model.cpp  [mod]
  |-- test_correctness.py  [+new]
  `-- tools/
      `-- server/
          |-- server.cpp  [mod]
          `-- utils.hpp  [mod]
```