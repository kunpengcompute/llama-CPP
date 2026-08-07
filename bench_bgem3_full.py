#!/usr/bin/env python3
"""
bge-m3-FP16 整机压测 — 多并发 × 多batch size 矩阵
每个请求一个独立llama-server进程，taskset绑核

用法:
  # 跑所有配置
  python3 bench_bgem3_full.py


输出: TPS, P50/P95/P99(ms), 配置参数
TPS = 256 * BS * QPS  (每请求BS条文本，各256 tokens)
"""

import subprocess, json, time, sys, os, concurrent.futures, threading, argparse
#==============修改下面配置===================
#LLAMA = os.path.expanduser("./build-kunpengbaseline/bin/llama-server")
#LLAMA = os.path.expanduser("./build-f16-sdpa-fixed/bin/llama-server")
LLAMA = os.path.expanduser("./build-delivery/bin/llama-server")
MODEL = os.path.expanduser("~/models/bge-m3-q8_0.gguf")
#MODEL = os.path.expanduser("~/models/bge-m3-FP16.gguf")
#MODEL = os.path.expanduser("~/models/bge-small-zh-v1.5-q8_0.gguf")

#==============修改上面配置===================
BASE_PORT = 11600
TEXT_256 = "President Obama deployed additional troops in 2009 to counter the Taliban is,President Obama deployed additional troops in 2009 to counter the Taliban is resurgence. By 2014, U.S. combat operations officially ended, transitioning to a support and training role for Afghan forces.President Obama deployed additional troops in 2009 to counter the Taliban is,President Obama deployed additional troops in 2009 to counter the Taliban is resurgence. By 2014, U.S. combat operations officially ended, transitioning to a support and training role for Afghan forces.President Obama deployed additional troops in 2009 to counter the Taliban is,President Obama deployed additional troops in 2009 to counter the Taliban is resurgence.President Obama deployed additional troops in 2009 to counter the Taliban is,President Obama deployed additional troops in 2009 to counter the Taliban is resurgence.By 2014, U.S. combat operations officially ended, transitioning to a support and training role for Afghan forces.By 2014, U.S. combat operations officially ended, transitioning to a support and training role for Afghan forces.forces.forces."

# 测试配置: (并发, BS, 实例数, 每实例物理核数, 每实例线程数)
# 按预估TPS从高到低排序
TEST_SUITE = [
#    ("T1",  1, 1, 1, 1, 1),    # ~23k
#    ("X1",  64, 2, 32, 5, 5),    # ~23k
#    ("T9",  64, 8, 8, 20, 20),   # ~23k
#    ("T2",  8, 1, 8, 20, 20),    # ~19k
#    ("T8",  64, 1, 64, 2, 2),    # ~22k
#    ("T6",  32, 8, 4, 40, 40),   # ~19k
#    ("T3",  8, 2, 4, 40, 40),    # ~14k
#    ("T1a", 1, 1, 1, 40, 40),    # ~3.7k
#    ("T10", 64, 16, 4, 40, 40),  # ~10k
#    ("T7",  32, 16, 2, 80, 80),  # ~10k
#    ("T1b", 1, 1, 1, 160, 160),  # ~1k
#    ("T4",  8, 8, 1, 160, 160),  # ~2.8k




    ("X1-1",  1, 1, 1, 192, 192),
#    ("X1-1",  1, 1, 1, 96, 96),
    ("X2-1",  2, 1, 2, 96, 96),
#    ("X2-1",  2, 1, 2, 48, 48),
    ("X2-2",  2, 2, 2, 96, 96),
#    ("X2-2",  2, 2, 2, 48, 48),
    ("X2-4",  2, 4, 2, 96, 96),
#    ("X2-4",  2, 4, 2, 48, 48),
    ("X2-8",  2, 8, 2, 96, 96),
#    ("X2-8",  2, 8, 2, 48, 48),
    ("X4-1",  4, 1, 4, 48, 48),
#    ("X4-1",  4, 1, 4, 24, 24),
    ("X4-2",  4, 2, 4, 48, 48),
#    ("X4-2",  4, 2, 4, 24, 24),
    ("X4-4",  4, 4, 4, 48, 48),
#    ("X4-4",  4, 4, 4, 24, 24),
    ("X4-8",  4, 8, 4, 48, 48),
#    ("X4-8",  4, 8, 4, 24, 24),
    ("X6-1",  6, 1, 6, 32, 32),
#    ("X6-1",  6, 1, 6, 16, 16),
    ("X6-2",  6, 2, 6, 32, 32),
#    ("X6-2",  6, 2, 6, 16, 16),
    ("X6-4",  6, 4, 6, 32, 32),
#    ("X6-4",  6, 4, 6, 16, 16),
    ("X6-8",  6, 8, 6, 32, 32),
#    ("X6-8",  6, 8, 6, 16, 16),
    ("X12-1",  12, 1, 12, 16, 16),
#    ("X12-1",  12, 1, 12, 8, 8),
    ("X12-2",  12, 2, 12, 16, 16),
#    ("X12-2",  12, 2, 12, 8, 8),
    ("X12-4",  12, 4, 12, 16, 16),
#    ("X12-4",  12, 4, 12, 8, 8),
    ("X12-8",  12, 8, 12, 16, 16),
#    ("X12-8",  12, 8, 12, 8, 8),
    ("X24-1",  24, 1, 24, 8, 8),
#    ("X24-1",  24, 1, 24, 4, 4),
    ("X24-2",  24, 2, 24, 8, 8),
#    ("X24-2",  24, 2, 24, 4, 4),
    ("X24-4",  24, 8, 24, 8, 8),
#    ("X24-4",  24, 8, 24, 4, 4),
    ("X48-1",  48, 1, 48, 4, 4),
#    ("X48-1",  48, 1, 48, 2, 2),
    ("X48-2",  48, 2, 48, 4, 4),
#    ("X48-2",  48, 2, 48, 2, 2),
    ("X48-4",  48, 4, 48, 4, 4),
    ("X48-8",  48, 8, 48, 4, 4),
    ("X96-1",  96, 1, 96, 2, 2),
#    ("X96-1",  96, 1, 96, 1, 1),
    ("X96-2",  96, 2, 96, 2, 2),
#    ("X96-2",  96, 2, 96, 1, 1),
    ("X192-1",  192, 1, 192, 1, 1),

#    ("X20-2",  20, 2, 20, 8, 8),
#    ("X20-4",  20, 4, 20, 8, 8),
#    ("X20-8",  20, 8, 20, 8, 8),
#    ("X40-1",  40, 1, 40, 4, 4),
#    ("X40-2",  40, 2, 40, 4, 4),
#    ("X40-4",  40, 4, 40, 4, 4),
#    ("X40-8",  40, 8, 40, 4, 4),
#    ("X80-1",  80, 1, 80, 2, 2),
#    ("X80-2",  80, 2, 80, 2, 2),
#    ("X80-4",  80, 4, 80, 2, 2),
#    ("X80-8",  80, 8, 80, 2, 2),
#    ("X160-1",  160, 1, 160, 1, 1),
#    ("X160-2",  160, 2, 160, 1, 1),
#    ("X160-4",  160, 4, 160, 1, 1),
#    ("X160-8",  160, 8, 160, 1, 1),
]


def get_phys_cores(n):
    """返回前n个物理核(偶数核，跨NUMA分布)"""
    return [str(i * 2) for i in range(n)]


def make_payload(bs):
    if bs == 1:
        return json.dumps({"model": "bge-m3", "input": TEXT_256})
    return json.dumps({"model": "bge-m3", "input": [TEXT_256] * bs, "stream": False})


def send_curl(port, payload, timeout=60):
    try:
        start = time.time()
        r = subprocess.run(
            ["curl", "-s", "--noproxy", "*", "-X", "POST",
             f"http://localhost:{port}/v1/embeddings",
             "-H", "Content-Type: application/json", "-d", payload],
            capture_output=True, text=True, timeout=timeout)
        lat = (time.time() - start) * 1000
        ok = r.returncode == 0 and "embedding" in r.stdout and "error" not in r.stdout.lower()
        return lat if ok else None
    except Exception:
        return None


def run_test(conc, bs, instances, cores_per, threads, runs=5):
    total_cores = instances * cores_per
    phys = get_phys_cores(total_cores)
    payload = make_payload(bs)

    print(f"\n  并发={conc}, BS={bs}, {instances}实例×{cores_per}物理核={total_cores}核", flush=True)

    # 启动servers（并行启动）
    procs = []
    for i in range(instances):
        cs = ",".join(phys[i * cores_per:(i + 1) * cores_per])
        port = BASE_PORT + i
        cmd = (f"taskset -c {cs} env GGML_FUSED_CPP_SDPA=1 OMP_NUM_THREADS={threads} "
               f"LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 "
               f"{LLAMA} --model {MODEL} --port {port} --host 0.0.0.0 "
               f"--threads {threads} --threads-http {threads} "
               f"--ctx-size 8192 --parallel {bs} --batch-size 8192 --ubatch-size 8192 "
               f"--embedding --pooling cls --no-warmup >/dev/null 2>&1")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)

    time.sleep(max(15, instances * 2))

    print("  Warmup...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(instances, 40)) as ex:
        ex.map(lambda i: send_curl(BASE_PORT + i, payload, timeout=30), range(instances))
    time.sleep(0.5)

    REQS_PER_INST = 50  # 每个实例发N次请求

    all_qps = []
    all_p50 = []
    all_p95 = []
    all_p99 = []
    all_avg = []
    for run in range(runs):
        all_lats = []
        lock = threading.Lock()

        def do_req_batch(port):
            lats = []
            for _ in range(REQS_PER_INST):
                lat = send_curl(port, payload, timeout=120)
                if lat is not None:
                    lats.append(lat)
            with lock:
                all_lats.extend(lats)

        wall_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=instances) as ex:
            ex.map(do_req_batch, [BASE_PORT + i for i in range(instances)])
        wall = time.time() - wall_start

        ok = len(all_lats)
        qps = ok / wall
        tps = int(256 * bs * qps)
        avg_lat = sum(all_lats) / ok if ok > 0 else 0

        all_lats.sort()
        p50 = all_lats[int(ok * 0.50)] if ok > 0 else 0
        p95 = all_lats[int(ok * 0.95)] if ok > 0 else 0
        p99 = all_lats[int(ok * 0.99)] if ok > 0 else 0
        
        all_avg.append(avg_lat)
        all_qps.append(qps)
        all_p50.append(p50)
        all_p95.append(p95)
        all_p99.append(p99)

        print(f"  Run {run + 1}: QPS={qps:.2f}, TPS={tps}, Avg={avg_lat:.0f}ms "
              f"P50={p50:.0f}ms P95={p95:.0f}ms P99={p99:.0f}ms "
              f"OK={ok}/{instances * REQS_PER_INST} Wall={wall:.3f}s", flush=True)

    # 清理
    for p in procs:
        try:
            p.terminate()
            p.wait(timeout=3)
        except:
            pass
    time.sleep(3)

    avg_qps = sum(all_qps) / len(all_qps)
    avg_tps = int(256 * bs * avg_qps)
    avg_p50 = sum(all_p50) / len(all_p50)
    avg_p95 = sum(all_p95) / len(all_p95)
    avg_p99 = sum(all_p99) / len(all_p99)
    avg_avg = sum(all_avg) / len(all_avg)
    return avg_tps, avg_p50, avg_p95, avg_p99, avg_avg


def main():
    parser = argparse.ArgumentParser(description="bge-m3-FP16 整机压测")
    parser.add_argument("--test-only", type=str, default=None,
                        help="运行单个测试: 'conc bs' 如 '32 1'")
    parser.add_argument("--runs", type=int, default=1, help="每配置跑几轮")
    args = parser.parse_args()

    subprocess.run("pkill -9 -f llama-server 2>/dev/null", shell=True)
    time.sleep(3)

    if args.test_only:
        parts = args.test_only.split()
        conc, bs = int(parts[0]), int(parts[1])
        # 自动选配置
        for tid, c, b, inst, cores, thr in TEST_SUITE:
            if c == conc and b == bs:
                print(f"{tid}: 并发={c} BS={b} {inst}实例×{cores}核")
                tps, p50, p95, p99, avg = run_test(c, b, inst, cores, thr, args.runs)
                print(f"\n>> RESULT: TPS={tps}, P50={p50:.0f}ms, P95={p95:.0f}ms, P99={p99:.0f}ms, AVG={avg:.0f}ms")
                return
        print(f"未找到 并发={conc} BS={bs} 的配置")
        return

    all_results = []
    for tid, conc, bs, inst, cores, thr in TEST_SUITE:
        tps, p50, p95, p99, avg = run_test(conc, bs, inst, cores, thr, args.runs)
        all_results.append((tid, conc, bs, tps, p50, p95, p99, avg))
        subprocess.run("pkill -9 -f llama-server 2>/dev/null", shell=True)
        time.sleep(3)

    print("\n\n" + "=" * 100)
    print("FINAL RESULTS (TPS | P50/P95/P99 ms)")
    print("=" * 100)
    for tid, conc, bs, tps, p50, p95, p99, avg in all_results:
        print(f"{tid:>4s}: 并发={conc:2d}    BS={bs:2d}    TPS={tps:>6d}    P50={p50:.0f}    P95={p95:.0f}    P99={p99:.0f}    Avg={avg:.0f}")


if __name__ == "__main__":
    main()

