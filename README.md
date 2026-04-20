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