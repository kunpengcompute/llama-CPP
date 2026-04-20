#!/usr/local/python3
# -*- encoding: utf-8 -*-
import sys
import time
import threading
import concurrent.futures
import requests
from openpyxl import Workbook

# 配置
MODEL_URL = "http://localhost:9180/v1/embeddings"

PAYLOAD = {
    "input": "query: The U.S. military operations in Afghanistan, primarily known as Operation Enduring Freedom, began in October 2001 following the September 11 attacks. The initial objective was to dismantle al-Qaeda and remove the Taliban regime from power, which had provided sanctuary to the terrorist group. Key phases of the conflict include:Initial Invasion 2001-2002: U.S. and coalition forces, along with the Northern Alliance, rapidly overthrew the Taliban government. Major cities like Kabul and Kandahar were secured, and al-Qaeda training camps were destroyed.Counterinsurgency and Nation-Building 2003-2014: The focus shifted to stabilizing Afghanistan, building democratic institutions, and training Afghan security forces. However, the Taliban regrouped and launched a resilient insurgency.Surge and Drawdown 2009-2014: President Obama deployed additional troops in 2009 to counter the Taliban is resurgence",
    "model": "bge-m3-FP16.gguf"
}

CONCURRENCY = 32  # 默认并发数
REQUESTS_PER_THREAD = 1  # 每个线程的请求数

# 统计变量（使用线程锁保护）
lock = threading.Lock()
total_requests = 0
successful_requests = 0
failed_requests = 0
total_time = 0.0
test_res = []

# 发送单个请求
def send_request():
    global total_requests, successful_requests, failed_requests, total_time

    start_time = time.time()
    try:
        response = requests.post(MODEL_URL, json=PAYLOAD, timeout=30)
        elapsed = time.time() - start_time

        with lock:
            total_requests += 1
            if response.status_code == 200:
                successful_requests += 1
                total_time += elapsed
            else:
                failed_requests += 1
                print(f"Request failed with status {response.status_code}: {response.text[:100]}")

    except Exception as e:
        with lock:
            total_requests += 1
            failed_requests += 1
        print(f"Request exception: {e}")

# 并发测试
def run_concurrency_test():
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(send_request) for _ in range(CONCURRENCY * REQUESTS_PER_THREAD)]
        concurrent.futures.wait(futures)
    end_time = time.time()
    
    # 防止除以0报错
    avg_latency_ms = (total_time / total_requests * 1000) if total_requests > 0 else 0
    duration = end_time - start_time
    throughput = successful_requests / duration if duration > 0 else 0

    print(
        "\nTest Results: "
        "Total Requests: {}, "
        "Successful Requests: {}, "
        "Failed Requests: {}, "
        "Total Time: {:.2f} seconds, "
        "Throughput: {:.2f} requests/second, "
        "Average Latency: {:.2f} ms".format(
            total_requests,
            successful_requests,
            failed_requests,
            duration,
            throughput,
            avg_latency_ms
        )
    )
    test_res.append({"Total Requests":total_requests,"Total Time":'{:.2f}'.format(duration), "Throughput":'{:.2f}'.format(throughput), "Average Latency":'{:.2f}'.format(avg_latency_ms)})

if __name__ == "__main__":
    # 创建一个新的 Excel 工作簿
    wb = Workbook()
    ws = wb.active  # 获取默认的工作表
    headers = [
        "Total Requests",
        "Total Time (seconds)",
        "Throughput (requests/second)",
        "Average Latency (ms)"
    ]
    ws.append(headers)  # 写入表头
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if len(first_arg.split(",")) > 0:
            for arg in first_arg.split(","):
                CONCURRENCY = int(arg)
                # 重置全局变量
                total_requests = 0
                successful_requests = 0
                failed_requests = 0
                total_time = 0
                run_concurrency_test()

        else:
            CONCURRENCY = int(first_arg)
            run_concurrency_test()

    # 写入多组数据
    for result in test_res:
        row = [
            result["Total Requests"],
            result["Total Time"],
            result["Throughput"],
            result["Average Latency"]
        ]
        ws.append(row)  # 写入一行数据
        print(
            "\nTest Results: "
            "Total Requests: {}, "
            "Total Time: {} seconds, "
            "Throughput: {} requests/second, "
            "Average Latency: {} ms".format(
                result["Total Requests"],
                result["Total Time"],
                result["Throughput"],
                result["Average Latency"]
            )
        )

    # 保存 Excel 文件
    wb.save("test_results.xlsx")
    print("Excel 文件已生成：test_results.xlsx")
