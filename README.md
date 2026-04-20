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
920B性能：
```shell
| model                          |       size |     params | backend    | threads | n_batch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | --------------: | -------------------: |
| bert 335M F16                  |   1.07 GiB |   566.70 M | CPU        |      16 |     256 |           pp256 |       1641.90 ± 6.83 |

build: 5a4c21944 (5590)

```


2. 端到端向llama-server发送embedding请求
```shell
python bge_embedding_server_test.py 1,2,4,8,16
```

920B性能：
```shell
Test Results: Total Requests: 1, Successful Requests: 1, Failed Requests: 0, Total Time: 0.15 seconds, Throughput: 6.88 requests/second, Average Latency: 144.54 ms

Test Results: Total Requests: 2, Successful Requests: 2, Failed Requests: 0, Total Time: 0.27 seconds, Throughput: 7.53 requests/second, Average Latency: 199.50 ms

Test Results: Total Requests: 4, Successful Requests: 4, Failed Requests: 0, Total Time: 0.53 seconds, Throughput: 7.49 requests/second, Average Latency: 333.85 ms

Test Results: Total Requests: 8, Successful Requests: 8, Failed Requests: 0, Total Time: 1.08 seconds, Throughput: 7.43 requests/second, Average Latency: 606.34 ms

Test Results: Total Requests: 16, Successful Requests: 16, Failed Requests: 0, Total Time: 2.14 seconds, Throughput: 7.46 requests/second, Average Latency: 1135.86 ms

Test Results: Total Requests: 1, Total Time: 0.15 seconds, Throughput: 6.88 requests/second, Average Latency: 144.54 ms

Test Results: Total Requests: 2, Total Time: 0.27 seconds, Throughput: 7.53 requests/second, Average Latency: 199.50 ms

Test Results: Total Requests: 4, Total Time: 0.53 seconds, Throughput: 7.49 requests/second, Average Latency: 333.85 ms

Test Results: Total Requests: 8, Total Time: 1.08 seconds, Throughput: 7.43 requests/second, Average Latency: 606.34 ms

Test Results: Total Requests: 16, Total Time: 2.14 seconds, Throughput: 7.46 requests/second, Average Latency: 1135.86 ms
Excel 文件已生成：test_results.xlsx
```