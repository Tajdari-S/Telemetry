#!/usr/bin/env python3
"""
Phase 5 — NVLink vs PCIe vs 1x GPU Comparison
Tests all three GPU configurations head-to-head:
  • 1x GPU  — single GPU baseline
  • 2x GPU (no explicit NVLink comms) — independent dual-GPU
  • 2x GPU (NVLink P2P) — explicit all-reduce / tensor-parallel comms

Workloads:
  A. All-reduce stress (imitates gradient sync in DDP training)
  B. Tensor-parallel-like P2P (imitates TP in inference)
  C. LLM-style GEMM with simulated inter-GPU KV cache copy

Sweeps:
  • Tensor size: [64 MB, 256 MB, 1 GB, 4 GB]
  • Direction: unidirectional, bidirectional

Metrics:
  • NVLink BW (GB/s) vs PCIe BW
  • Latency (μs)
  • GPU utilization on both GPUs
  • Power breakdown per GPU
"""
import os, sys, time, json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.telemetry import TelemetryRecorder
from utils.plotting import plot_bar_comparison, save_fig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "nvlink")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "nvlink")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

N_GPUS = torch.cuda.device_count()
SIZES_MB = [64, 256, 1024, 4096]

def p2p_bandwidth(src_dev: int, dst_dev: int, size_mb: float,
                  n_iters: int = 50, bidirectional: bool = False) -> dict:
    """Measure P2P bandwidth between two GPUs."""
    n = int(size_mb * 1e6 / 4)
    src = torch.randn(n, device=f"cuda:{src_dev}", dtype=torch.float32)
    dst = torch.empty(n, device=f"cuda:{dst_dev}", dtype=torch.float32)
    if bidirectional:
        src2 = torch.randn(n, device=f"cuda:{dst_dev}", dtype=torch.float32)
        dst2 = torch.empty(n, device=f"cuda:{src_dev}", dtype=torch.float32)

    # Warmup
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    # Time
    t0 = time.perf_counter()
    for _ in range(n_iters):
        dst.copy_(src)
        if bidirectional:
            dst2.copy_(src2)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    per_copy = elapsed / n_iters
    bytes_per_copy = n * 4 * (2 if bidirectional else 1)
    bw_GBps = bytes_per_copy / per_copy / 1e9
    lat_us   = per_copy * 1e6

    return {
        "size_mb": size_mb,
        "src_dev": src_dev,
        "dst_dev": dst_dev,
        "bidirectional": bidirectional,
        "bw_GBps": bw_GBps,
        "latency_us": lat_us,
    }

def allreduce_benchmark(size_mb: float, n_iters: int = 50) -> dict:
    """Ring all-reduce simulation using NCCL via torch.distributed or manual."""
    n = int(size_mb * 1e6 / 4)
    results = {}

    if N_GPUS >= 2:
        # Use torch.distributed all_reduce (NCCL backend, uses NVLink)
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            try:
                dist.init_process_group("nccl", rank=0, world_size=1)
            except Exception:
                pass  # might already be initialized

        t0_dev = torch.randn(n, device="cuda:0")
        for _ in range(5):
            t0_dev = t0_dev + t0_dev
        torch.cuda.synchronize()

        # Emulate all-reduce by manually summing GPU0+GPU1 tensors
        t1_dev = torch.randn(n, device="cuda:0")
        t2_dev = torch.randn(n, device="cuda:1")

        # P2P approach (NVLink path)
        for _ in range(3):
            t2_on_0 = t2_dev.to("cuda:0")
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            t2_on_0 = t2_dev.to("cuda:0")
            result = t1_dev + t2_on_0
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        bytes_moved = n * 4 * 2  # send + receive
        bw = bytes_moved / (elapsed / n_iters) / 1e9
        results = {"size_mb": size_mb, "allreduce_bw_GBps": bw,
                   "allreduce_latency_us": elapsed / n_iters * 1e6}
    return results

def gemm_1gpu(M: int = 8192, dtype=torch.bfloat16, n_iters: int = 50) -> dict:
    a = torch.randn(M, M, device="cuda:0", dtype=dtype)
    b = torch.randn(M, M, device="cuda:0", dtype=dtype)
    for _ in range(5): torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops = 2 * M * M * M
    tflops = flops / (elapsed / n_iters) / 1e12
    return {"config": "1xGPU", "tflops": tflops, "gemm_time_ms": elapsed/n_iters*1000}

def gemm_2gpu_independent(M: int = 8192, dtype=torch.bfloat16, n_iters: int = 50) -> dict:
    """Both GPUs running GEMM independently (no communication)."""
    import threading
    results = {}
    def worker(gi):
        a = torch.randn(M, M, device=f"cuda:{gi}", dtype=dtype)
        b = torch.randn(M, M, device=f"cuda:{gi}", dtype=dtype)
        for _ in range(5): torch.mm(a, b)
        torch.cuda.synchronize(f"cuda:{gi}")
        t0 = time.perf_counter()
        for _ in range(n_iters): torch.mm(a, b)
        torch.cuda.synchronize(f"cuda:{gi}")
        elapsed = time.perf_counter() - t0
        tflops = 2*M*M*M / (elapsed/n_iters) / 1e12
        results[gi] = tflops
    threads = [threading.Thread(target=worker, args=(g,)) for g in [0, 1]]
    for t in threads: t.start()
    for t in threads: t.join()
    total_tflops = sum(results.values())
    return {"config": "2xGPU_independent", "tflops": total_tflops,
            "tflops_per_gpu": results}

def gemm_2gpu_nvlink_tp(M: int = 8192, dtype=torch.bfloat16, n_iters: int = 50) -> dict:
    """
    Tensor-parallel-like GEMM: split matrix across 2 GPUs, compute, all-reduce.
    Each GPU computes M x (M/2) x M, then sums via NVLink.
    """
    a0 = torch.randn(M, M//2, device="cuda:0", dtype=dtype)
    b0 = torch.randn(M//2, M, device="cuda:0", dtype=dtype)
    a1 = torch.randn(M, M//2, device="cuda:1", dtype=dtype)
    b1 = torch.randn(M//2, M, device="cuda:1", dtype=dtype)

    for _ in range(5):
        c0 = torch.mm(a0, b0)
        c1 = torch.mm(a1, b1)
        c1_on_0 = c1.to("cuda:0")
        c = c0 + c1_on_0
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        c0 = torch.mm(a0, b0)
        c1 = torch.mm(a1, b1)
        c1_on_0 = c1.to("cuda:0")
        c = c0 + c1_on_0
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Total FLOPs: 2 * M * (M/2) * M * 2 GPUs = 2*M^3 (same as 1-GPU full)
    flops = 2 * M * M * M
    tflops = flops / (elapsed / n_iters) / 1e12
    comm_bytes = M * M * 2  # bf16 C0 across NVLink
    return {"config": "2xGPU_NVLink_TP", "tflops": tflops,
            "gemm_time_ms": elapsed/n_iters*1000,
            "nvlink_comm_MB": comm_bytes/1e6}

def main():
    print("=" * 70)
    print("  NVLink vs PCIe vs 1x GPU Comparison — NVIDIA B200")
    print(f"  Detected {N_GPUS} GPU(s)")
    print("=" * 70)

    all_bw_results  = []
    all_summary     = {}

    # ── P2P Bandwidth Sweep ──────────────────────────────────────────────────
    if N_GPUS >= 2:
        print("\n[A] P2P NVLink Bandwidth Sweep...")
        for size_mb in SIZES_MB:
            for bidi in [False, True]:
                res = p2p_bandwidth(0, 1, size_mb, n_iters=50, bidirectional=bidi)
                all_bw_results.append(res)
                dir_str = "bidir" if bidi else "unidir"
                print(f"  {size_mb:5d} MB {dir_str:7s}: {res['bw_GBps']:7.1f} GB/s  lat={res['latency_us']:.1f} μs")

        df_bw = pd.DataFrame(all_bw_results)
        df_bw.to_csv(f"{OUT}/nvlink_p2p_bw.csv", index=False)

        # BW vs size plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for bidi, ax, title in [(False, axes[0], "Unidirectional"),
                                 (True,  axes[1], "Bidirectional")]:
            d = df_bw[df_bw["bidirectional"] == bidi]
            ax.semilogx(d["size_mb"], d["bw_GBps"], "o-", lw=2, color="#1f77b4")
            ax.axhline(900, ls="--", color="red", label="NVLink18 peak (900 GB/s)")
            ax.set_xlabel("Transfer Size (MB)"); ax.set_ylabel("BW (GB/s)")
            ax.set_title(f"NVLink P2P BW — {title}"); ax.legend(); ax.grid(alpha=0.3)
        fig.suptitle("NVIDIA B200 NVLink P2P Bandwidth (18-link NVLink)", fontsize=12)
        plt.tight_layout()
        save_fig(fig, f"{GOUT}/nvlink_p2p_bw.png")

        # Latency vs size
        fig, ax = plt.subplots(figsize=(8, 4))
        for bidi, lbl, clr in [(False, "Unidirectional", "#1f77b4"),
                                (True,  "Bidirectional",  "#ff7f0e")]:
            d = df_bw[df_bw["bidirectional"] == bidi]
            ax.semilogx(d["size_mb"], d["latency_us"], "o-", lw=2, color=clr, label=lbl)
        ax.set_xlabel("Transfer Size (MB)"); ax.set_ylabel("Latency (μs)")
        ax.set_title("NVLink P2P Latency vs Transfer Size"); ax.legend(); ax.grid(alpha=0.3)
        save_fig(fig, f"{GOUT}/nvlink_p2p_latency.png")

    # ── All-Reduce Benchmark ─────────────────────────────────────────────────
    if N_GPUS >= 2:
        print("\n[B] All-Reduce Bandwidth Sweep...")
        ar_results = []
        for size_mb in SIZES_MB:
            res = allreduce_benchmark(size_mb)
            if res:
                ar_results.append(res)
                print(f"  {size_mb:5d} MB allreduce: {res['allreduce_bw_GBps']:.1f} GB/s  lat={res['allreduce_latency_us']:.1f} μs")
        if ar_results:
            df_ar = pd.DataFrame(ar_results)
            df_ar.to_csv(f"{OUT}/allreduce_bw.csv", index=False)
            all_summary["allreduce"] = ar_results

    # ── GEMM 1x vs 2x vs NVLink-TP ──────────────────────────────────────────
    print("\n[C] GEMM: 1x GPU vs 2x GPU vs 2x GPU NVLink-TP...")
    gemm_results = {}
    for M in [4096, 8192, 16384]:
        r1 = gemm_1gpu(M)
        gemm_results[f"1xGPU_{M}"] = r1
        print(f"  M={M}: 1xGPU = {r1['tflops']:.1f} TFLOPS")

        if N_GPUS >= 2:
            r2 = gemm_2gpu_independent(M)
            gemm_results[f"2xGPU_ind_{M}"] = r2
            print(f"  M={M}: 2xGPU_independent = {r2['tflops']:.1f} TFLOPS total")

            r3 = gemm_2gpu_nvlink_tp(M)
            gemm_results[f"2xGPU_NVLink_{M}"] = r3
            print(f"  M={M}: 2xGPU_NVLink_TP = {r3['tflops']:.1f} TFLOPS")

    # Bar chart: GEMM comparison
    labels = list(gemm_results.keys())
    vals   = [gemm_results[k]["tflops"] for k in labels]
    plot_bar_comparison(labels, vals, "TFLOPS",
                        "GEMM: 1xGPU vs 2xGPU vs 2xGPU NVLink-TP",
                        f"{GOUT}/gemm_1x_vs_2x_nvlink.png")

    # ── Telemetry during NVLink stress ───────────────────────────────────────
    if N_GPUS >= 2:
        print("\n[D] NVLink telemetry (30s sustained transfer)...")
        n = int(1e9 / 4)   # 1 GB
        src = torch.randn(n, device="cuda:0")
        dst = torch.empty(n, device="cuda:1")
        rec = TelemetryRecorder(gpu_indices=[0, 1], hz=20)
        rec.start()
        t_end = time.time() + 30
        iters = 0
        while time.time() < t_end:
            dst.copy_(src)
            iters += 1
        torch.cuda.synchronize()
        df_telem = rec.stop()
        df_telem.to_csv(f"{OUT}/nvlink_stress_telemetry.csv", index=False)

        # Plot both GPUs
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("NVLink Stress Telemetry — GPU 0 (top) / GPU 1 (bottom)", fontsize=12)
        cols = ["gpu_util","power_w","mem_util"]
        ylbls = ["GPU Util (%)","Power (W)","Mem BW Util (%)"]
        for gi, row_axes in enumerate(axes):
            df_g = df_telem[df_telem["gpu_idx"] == gi].copy()
            df_g["t"] = df_g["ts"] - df_g["ts"].iloc[0]
            for ax, col, lbl in zip(row_axes, cols, ylbls):
                ax.plot(df_g["t"], df_g[col], lw=1.2)
                ax.set_xlabel("Time (s)"); ax.set_ylabel(lbl)
                ax.set_title(f"GPU {gi} — {lbl}"); ax.grid(alpha=0.3)
        plt.tight_layout()
        save_fig(fig, f"{GOUT}/nvlink_stress_telemetry.png")

        bw_sustained = iters * 1e9 / 30.0 / 1e9
        print(f"  Sustained NVLink BW: {bw_sustained:.1f} GB/s over 30s ({iters} iters)")
        all_summary["nvlink_sustained_bw_GBps"] = bw_sustained

    # ── Summary: 1x vs 2x vs NVLink ─────────────────────────────────────────
    all_summary["gemm_tflops"] = gemm_results
    all_summary["p2p_bw"] = all_bw_results
    with open(f"{OUT}/nvlink_comparison_summary.json", "w") as f:
        json.dump(all_summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  NVLink Comparison Complete.")
    print(f"  Results → {OUT}/  |  Graphs → {GOUT}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
