# Technical Report

## Accuracy Verification

 Before accuracy verification, you must use `patch/baseline-bug-fixed.patch` to fix the baseline issues of the official llama.cpp repository (commit `3ac67535c86`).

 The procedure is as follows.

 1. Switch the source code to the version corresponding to the official llama.cpp (commit `3ac67535c86`).
 2. Execute `git apply baseline-bug-fixed.patch` to fix the baseline issues.
 3. Compile this version and use it for accuracy verification.

 ```shell
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

>![icon-note](./public_sys-resources/icon-note.gif)**NOTE:**
>
>- Set the llama servers following `servers` in the format of `<server_name>=<server url>`. Multiple servers are separated by spaces.
>- The number of server-workers equals the number of servers above; use one worker for each server being tested.
>- Change the path following `cmteb-root` to your own C-MTEB dataset path.
>- If a task's score is `NaN`, you can run that task again.

C-MTEB accuracy result reference.

| Task                    | F16-baseline-fixed | F16-opt    | Q80-opt    | F16 Accuracy Loss | Q80 Accuracy Loss |
| ----------------------- | ------------------ | ---------- | ---------- | ----------------- | ----------------- |
| AFQMC                   | 33.42%             | 33.42%     | 33.37%     | 0.00%             | -0.04%            |
| ATEC                    | 43.71%             | 43.71%     | 43.71%     | 0.00%             | 0.00%             |
| BQ                      | 58.14%             | 58.13%     | 58.06%     | -0.01%            | -0.08%            |
| CLSClusteringS2S.v2     | 34.68%             | 34.83%     | 34.39%     | 0.14%             | -0.29%            |
| Cmnli                   | 69.06%             | 69.09%     | 69.04%     | 0.04%             | -0.01%            |
| IFlyTek                 | 45.75%             | 45.66%     | 45.65%     | -0.09%            | -0.10%            |
| JDReview                | 80.04%             | 79.83%     | 79.98%     | -0.21%            | -0.06%            |
| LCQMC                   | 72.21%             | 72.21%     | 72.17%     | 0.00%             | -0.04%            |
| MultilingualSentiment   | 62.96%             | 62.94%     | 62.95%     | -0.02%            | -0.01%            |
| Ocnli                   | 62.10%             | 62.10%     | 62.05%     | 0.00%             | -0.05%            |
| OnlineShopping          | 84.92%             | 84.97%     | 84.95%     | 0.05%             | 0.03%             |
| PAWSX                   | 9.76%              | 9.77%      | 9.71%      | 0.01%             | -0.06%            |
| QBQTC                   | 35.17%             | 35.16%     | 35.10%     | 0.00%             | -0.06%            |
| STSB                    | 80.67%             | 80.66%     | 80.69%     | 0.00%             | 0.02%             |
| TNews                   | 48.68%             | 48.69%     | 48.72%     | 0.00%             | 0.03%             |
| ThuNewsClusteringP2P.v2 | 55.93%             | 55.55%     | 55.61%     | -0.38%            | -0.32%            |
| ThuNewsClusteringS2S.v2 | 49.51%             | 49.35%     | 49.05%     | -0.16%            | -0.46%            |
| Waimai                  | 83.14%             | 83.14%     | 83.14%     | 0.00%             | 0.00%             |
| Average                 | **56.10%**         | **56.07%** | **56.02%** | **-0.03%**        | **-0.08%**        |

## Performance Verification

### Single-Core Verification

```shell
taskset -c 128 env  GGML_FUSED_CPP_SDPA=1 GGML_TOTAL_THREADS=1 OMP_NUM_THREADS=1 ./build-delivery/bin/llama-bench -m /path/to/bge-m3-FP16.gguf -p 512 -b 512 -n 0 -t 1 -r 1
```

#### Parameter description

|Parameter|Description|
|---|---|
|GGML_FUSED_CPP_SDPA=1|Enables the SDPA feature.|
|GGML_TOTAL_THREADS=1|Indicates a single thread.|
|OMP_NUM_THREADS=1|Indicates a single OMP thread.|
|m|Specifies the model path.|
|p|Sets the input length of the prompt processing (prefill) phase to 512 tokens.|
|b|Sets the logical batch size to 512. In CPU-only inference with an input of 512 tokens, this typically means processing 512 tokens as a complete batch.|
|n|Sets the number of generated tokens to 0. This means completely skipping the text generation (decode) phase and testing only the throughput of the prefill phase.|
|t|Sets the number of llama-bench worker threads to 1.|
|r|Sets the number of repetitions per test to 1.|

### Full-System Verification

```shell
vim bench_bgem3_full.py   
python bench_bgem3_full.py   
```

### Optimized Performance Reference

#### Reference for Optimized Performance on New Kunpeng 920 Processor Models (2.9 GHz, 80 Cores × 2 Sockets × 2 NUMA Nodes), with 16 × 64 GB Memory @ 4800 MT/s

**Single-core performance after optimization:**

| Model             | Precision | Parameter Count | Model Size | Baseline (t/s) | Optimized (t/s) | Improvement Ratio | Performance Gain |
| :---------------- | :-------- | --------------: | ---------: | -------------: | --------------: | ----------------: | --------------: |
| bge-m3            | FP16      |       566.70 M  |   1.07 GiB |          21.36 |      **216.99** |        **10.16x** |   **+915.68%**  |
| bge-m3            | Q8_0      |       566.70 M  | 598.63 MiB |          55.01 |      **220.85** |         **4.01x** |   **+301.47%**  |
| bge-small-zh-v1.5 | FP16      |        23.69 M  |  45.24 MiB |         488.32 |     **4753.50** |         **9.73x** |   **+873.45%**  |
| bge-small-zh-v1.5 | Q8_0      |        23.69 M  |  24.82 MiB |        1024.32 |     **4624.93** |         **4.52x** |   **+351.52%**  |

**Full-system performance after optimization.**

Explanation of evaluation metrics.

- `TPS`: Indicates tokens per second.
- `P50`/`P99`/`Avg. Latency`: Based on the script configuration, each instance sends 50 requests by default. These values represent the mean P50/P99/average latency of those requests, in milliseconds.
- `Avg. Latency Per Sentence`: The batch size is 1 by default. For scenarios where the batch size is larger than 1, this value is calculated as `Avg. Latency/BS`.

After FP16 optimization, the optimal TPS is 31,097. Before optimization, the optimal TPS is 3,352.

| Instance Qty. | Core Qty./Instance | TPS | P50 | P99 | Avg. Latency | Avg. Latency Per Sentence |
| --------: | :------------- | ---------: | ---: | ---: | ---: | ------------------: |
| 1 | 160 | 1,080 | 242 | 280 | 237 | |
| 2 | 80 | 2,357 | 216 | 314 | 215 | |
| 4 | 40 (BS=1) | 13,036 | 78 | 88 | 78 | |
| 4 | 40 (BS=2) | 15,162 | 132 | 152 | 132 | 66 |
| 4 | 40 (BS=4) | 15,409 | 262 | 273 | 261 | 65.25 |
| 4 | 40 (BS=8) | 20,118 | 398 | 416 | 399 | 49.875 |
| 8 | 20 | 19,171 | 105 | 114 | 105 | |
| 8 | 20 (BS=8) | 24,939 | 655 | 666 | 654 | 81.75 |
| 16 | 10 | 22,095 | 182 | 204 | 182 | |
| 16 | 10 (BS=8) | 26,700 | 1201 | 1266 | 1204 | 150.5 |
| 20 | 8 (BS=1) | 24,727 | 202 | 222 | 202 | |
| 20 | 8 (BS=4) | 28,061 | 714 | 738 | 713 | 178.25 |
| 40 | 4 (BS=1) | 27,930 | 356 | 379 | 357 | |
| 40 | 4 (BS=2) | 30,515 | 663 | 686 | 664 | 332 |
| 40 | 4 (BS=4) | 30,276 | 1338 | 1370 | 1338 | 334.5 |
| 80 | 2 (BS=1) | 29,522 | 676 | 733 | 678 | |
| 80 | 2 (BS=2) | **31,097** | 1296 | 1344 | 1298 | |
| 80 | 2 (BS=4) | 29,451 | 2709 | 2807 | 2711 | |
| 160 | 1 | 30,212 | 1330 | 1428 | 1332 | |

**Q80 performance after optimization.**

After Q80 optimization, the optimal TPS is 32,148. Before optimization, the optimal TPS is 9,952.

| Instance Qty. | Core Qty./Instance | TPS | P50 | P99 | Avg. Latency | Avg. Latency Per Sentence |
| --------: | :------------- | ---: | ---: | ---: | ---: | ------------------: |
| 1 | 160 | 824 | 309 | 396 | 310 | |
| 2 | 80 | 2,010 | 249 | 363 | 240 | |
| 4 | 40 (BS=1) | 7,622 | 123 | 140 | 116 | |
| 4 | 40 (BS=2) | 9,672 | 202 | 215 | 193 | 96.5 |
| 4 | 40 (BS=4) | 11,229 | 361 | 370 | 361 | 90.25 |
| 4 | 40 (BS=8) | 12,557 | 636 | 658 | 634 | 79.25 |
| 8 | 20 | 13,292 | 152 | 160 | 152 | |
| 8 | 20 (BS=8) | 16,099 | 932 | 1031 | 953 | 119.125 |
| 16 | 10 | 13,808 | 255 | 373 | 268 | |
| 16 | 10 (BS=8) | 22,850 | 1374 | 1463 | 1390 | 173.75 |
| 20 | 8 (BS=1) | 27,487 | 183 | 190 | 183 | |
| 20 | 8 (BS=4) | 29,773 | 684 | 696 | 684 | 171 |
| 40 | 4 (BS=1) | 29,850 | 333 | 352 | 334 | |
| 40 | 4 (BS=2) | 31,175 | 643 | 662 | 643 | 321.5 |
| 40 | 4 (BS=4) | 31,503 | 1287 | 1323 | 1289 | 322.25 |
| 80 | 2 (BS=1) | 31,084 | 642 | 676 | 643 | |
| 80 | 2 (BS=2) | **32,148** | 1255 | 1288 | 1255 | |
| 80 | 2 (BS=4) | 31,826 | 2542 | 2602 | 2543 | |
| 160 | 1 | 31,468 | 1267 | 1344 | 1268 | |

#### Reference for Optimized Performance on Kunpeng 950 Processors (2.3 GHz, 96 Cores × 2 Sockets × 2 NUMA Nodes), with 12 × 96 GB Memory @ 6400 MT/s

**Comparison with the 9654 x86 processor (3.7 GHz, 96 cores × 2 sockets × 1 NUMA node), with 24 × 32 GB Memory @ 4800 MT/s.**

| Model | Test Item | 950 (Before Optimization) | 9654 (Before Optimization) | 950 (After Optimization) | Improvement Ratio (vs. 9654) |
| :----- | :------ | ---------: | ----------: | -----------: | -------------: |
| bge-small-zh-v1.5 | 16 physical cores | 12932.70 | 14803.95 | **45252.71** | **3.03x** |
| bge-small-zh-v1.5 | Single core | 901.60 | 1357.02 | **4266.47** | **3.14x** |
| bge-m3 | 16 physical cores | 643.30 | 844.23 | **2591.14** | **3.06x** |
| bge-m3 | Single core | 42.82 | 62.51 | **195.30** | **3.12x** |
| MemReader-4B | 16 physical cores-prefill | 59.66 | 87.53 | **234.80** | **2.68x** |
| MemReader-4B | 16 physical cores-decode | 15.95 | 11.38 | **16.36** | **1.43x** |
| toolcall-0.6b | 16 physical cores-prefill | 474.83 | 571.55 | **1467.07** | **2.57x** |
| toolcall-0.6b | 16 physical cores-decode | 85.55 | 58.58 | **89.03** | **1.51x** |

**Single-core performance after optimization (vs. baseline).**

| Model | Precision | Parameter Count | Model Size | Baseline (t/s) | Optimized (t/s) | Improvement Ratio | Performance Gain |
| :---------------- | :---- | -------: | ---------: | -------------: | --------------: | ---------: | -----------: |
| bge-m3 | FP16 | 566.70 M | 1.07 GiB | 39.03 | **193.14** | **4.95x** | **+394.85%** |
| bge-m3 | Q8_0 | 566.70 M | 598.63 MiB | 63.55 | **286.29** | **4.50x** | **+350.50%** |
| bge-small-zh-v1.5 | FP16 | 23.69 M | 45.24 MiB | 760.05 | **4175.56** | **5.49x** | **+449.38%** |
| bge-small-zh-v1.5 | Q8_0 | 23.69 M | 24.82 MiB | 1116.75 | **5702.74** | **5.11x** | **+410.66%** |

**Full-system performance after optimization.**

After FP16 optimization, the **optimal TPS is 30,037**. Before optimization, the optimal TPS is 7,758.

| Instance Qty. | Core Qty./Instance |        TPS |  P50 |  P99 |  Avg. Latency | Avg. Latency Per Sentence |
| ---: | :--------- | ---------: | ---: | ---: | ---: | -----------: |
|    1 | 192        |      3,050 |   64 |  494 |   84 |              |
|    2 | 96         |      6,362 |   65 |  349 |   78 |              |
|    2 | 96 (BS=2)  |      6,967 |  112 |  421 |  147 |         73.5 |
|    2 | 96 (BS=4)  |      6,797 |  263 |  581 |  285 |        71.25 |
|    2 | 96 (BS=8)  |     10,510 |  366 |  651 |  376 |           47 |
|    4 | 48         |      8,940 |   73 |  307 |   97 |              |
|    4 | 48 (BS=2)  |     12,355 |  145 |  407 |  150 |           75 |
|    4 | 48 (BS=4)  |     12,439 |  304 |  492 |  314 |         78.5 |
|    4 | 48 (BS=8)  |     16,008 |  500 |  663 |  493 |       61.625 |
|    6 | 32         |     15,141 |   84 |  301 |   87 |              |
|    6 | 32 (BS=2)  |     19,976 |  147 |  227 |  148 |           74 |
|    6 | 32 (BS=4)  |     19,851 |  290 |  463 |  285 |        71.25 |
|    6 | 32 (BS=8)  |     20,704 |  552 |  889 |  544 |           68 |
|   12 | 16 (BS=2)  |     24,525 |  231 |  276 |  233 |              |
|   12 | 16 (BS=4)  |     24,557 |  469 |  598 |  462 |        115.5 |
|   24 | 8          |     28,956 |  211 |  222 |  411 |              |
|   48 | 4          | **30,037** |  404 |  417 |  404 |              |
|   96 | 2          |     29,635 |  816 |  852 |  801 |              |
|  192 | 1          |     25,995 | 1685 | 2080 | 1731 |              |

**Q80 performance after optimization.**

After optimization, the **optimal TPS is 43,712**. Before optimization, the optimal TPS is 13,257.

| Instance Qty. | Core Qty./Instance | TPS | P50 | P99 | Avg. Latency | Avg. Latency Per Sentence |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | 192 | 3,317 | 61 | 431 | 77 | 77 |
| 2 | 96 (BS=1) | 5,891 | 57 | 306 | 80 | 80 |
| 2 | 96 (BS=2) | 7,423 | 101 | 407 | 123 | 61.5 |
| 2 | 96 (BS=4) | 10,231 | 186 | 386 | 198 | 49.5 |
| 2 | 96 (BS=8) | 10,974 | 327 | 589 | 355 | 44.38 |
| 4 | 48 (BS=1) | 11,915 | 79 | 265 | 83 | 83 |
| 4 | 48 (BS=2) | 13,183 | 143 | 338 | 145 | 72.5 |
| 4 | 48 (BS=4) | 14,135 | 267 | 488 | 281 | 70.25 |
| 4 | 48 (BS=8) | 16,151 | 479 | 691 | 485 | 60.63 |
| 6 | 32 (BS=1) | 21,159 | 66 | 98 | 65 | 65 |
| 6 | 32 (BS=2) | 24,026 | 117 | 161 | 115 | 57.5 |
| 6 | 32 (BS=4) | 24,903 | 228 | 286 | 221 | 55.25 |
| 6 | 32 (BS=8) | 26,259 | 402 | 546 | 410 | 51.25 |
| 12 | 16 (BS=1) | 29,149 | 94 | 123 | 95 | 95 |
| 12 | 16 (BS=2) | 33,052 | 173 | 218 | 173 | 86.5 |
| 12 | 16 (BS=4) | 30,276 | 360 | 430 | 361 | 90.25 |
| 12 | 16 (BS=8) | 30,664 | 698 | 867 | 705 | 88.13 |
| 24 | 8 (BS=1) | 40,271 | 144 | 159 | 144 | 144 |
| 24 | 8 (BS=2) | 43,667 | 277 | 290 | 276 | 138 |
| 24 | 8 (BS=8) | 31,339 | 1,353 | 1,633 | 1,362 | 170.25 |
| 48 | 4 (BS=1) | **43,712** | 271 | 287 | 272 | 272 |
| 48 | 4 (BS=2) | 43,373 | 548 | 572 | 546 | 273 |
| 96 | 2 (BS=1) | 43,648 | 549 | 573 | 549 | 549 |
| 96 | 2 (BS=2) | 34,825 | 1,131 | 1,475 | 1,245 | 622.5 |
| 192 | 1 | 34,383 | 1,157 | 1,500 | 1,254 | 1,254.00 |

## Change History

| Version | Date | Description |
| :--- | :--- | :--- |
| 01 | 2026-09-30 | This is the first official release. |
