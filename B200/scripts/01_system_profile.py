#!/usr/bin/env python3
"""
Phase 1 — B200 System Profiling
Measures:
  • HBM sequential and random-access bandwidth
  • L1/L2 cache effective bandwidth and latency
  • TLB stress (large-stride access)
  • SM throughput (FP32/TF32/FP16/BF16/FP8 GEMM)
  • Tensor Core throughput (GEMM)
  • Maximum sustained power
  • Roofline parameters
Saves JSON summary + plots to results/system_profile/
"""
import os, sys, json, time, math
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils.roofline import B200_SPECS, DTYPE_PEAK, plot_roofline, MEM_BW_TBps
from utils.telemetry import TelemetryRecorder
from utils.plotting import plot_bar_comparison, save_fig

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "system_profile")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "system_profile")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

def warmup(n=5):
    x = torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16)
    for _ in range(n):
        torch.mm(x, x)
    torch.cuda.synchronize()

def timed(fn, n_iters=20, warmup_iters=5):
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters  # seconds per iter

# ───────────────────────────────────────────────────────────────────────────
# 1. HBM Sequential Bandwidth
# ───────────────────────────────────────────────────────────────────────────
def bench_hbm_bw(size_gb: float = 4.0, dtype=torch.float32):
    """Copy kernel measures device→device bandwidth."""
    n = int(size_gb * 1e9 / dtype.itemsize)
    src = torch.ones(n, device=DEVICE, dtype=dtype)
    dst = torch.empty(n, device=DEVICE, dtype=dtype)

    def copy_fn():
        dst.copy_(src)

    elapsed = timed(copy_fn, n_iters=10, warmup_iters=3)
    bytes_moved = 2 * n * dtype.itemsize   # read + write
    bw_TBps = bytes_moved / elapsed / 1e12
    return bw_TBps

# ───────────────────────────────────────────────────────────────────────────
# 2. HBM Random Access Bandwidth
# ───────────────────────────────────────────────────────────────────────────
def bench_random_access(size_gb: float = 0.5, dtype=torch.float32):
    """Gather with random indices — worst-case random access pattern."""
    n = int(size_gb * 1e9 / dtype.itemsize)
    data  = torch.randn(n, device=DEVICE, dtype=dtype)
    idx   = torch.randint(0, n, (n,), device=DEVICE)
    out   = torch.empty(n, device=DEVICE, dtype=dtype)

    def fn():
        out.copy_(data[idx])

    elapsed = timed(fn, n_iters=10, warmup_iters=3)
    bytes_moved = n * dtype.itemsize * 2
    bw_TBps = bytes_moved / elapsed / 1e12
    return bw_TBps

# ───────────────────────────────────────────────────────────────────────────
# 3. Cache Hierarchy — Effective BW vs Working Set
# ───────────────────────────────────────────────────────────────────────────
def bench_cache_hierarchy():
    """
    Sweep working-set size from 32 KB up to 4 GB.
    Sequential read bandwidth reveals L1→L2→HBM transitions.
    """
    sizes_kb = [32, 64, 128, 256, 512, 1024,
                4*1024, 16*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024]
    results = []
    for sz_kb in sizes_kb:
        n = (sz_kb * 1024) // 4   # float32 elements
        if n < 1:
            continue
        buf = torch.ones(n, device=DEVICE, dtype=torch.float32)
        out = torch.empty(n, device=DEVICE, dtype=torch.float32)
        def fn():
            out.copy_(buf)
        elapsed = timed(fn, n_iters=30, warmup_iters=5)
        bw = 2 * n * 4 / elapsed / 1e12
        results.append({"size_kb": sz_kb, "bw_TBps": bw})
    return results

# ───────────────────────────────────────────────────────────────────────────
# 4. TLB Stress — Strided Access
# ───────────────────────────────────────────────────────────────────────────
def bench_tlb(strides_bytes=[64, 512, 4096, 65536, 2097152]):
    """
    Measure BW with different strides. TLB misses degrade BW at large strides.
    Uses 512 MB buffer, gathers with stride pattern.
    """
    size_bytes = 512 * 1024 * 1024
    n = size_bytes // 4
    buf = torch.ones(n, device=DEVICE, dtype=torch.float32)
    results = []
    for stride_b in strides_bytes:
        stride_elem = max(1, stride_b // 4)
        indices = torch.arange(0, n, stride_elem, device=DEVICE)
        out = torch.empty(len(indices), device=DEVICE, dtype=torch.float32)
        def fn():
            out.copy_(buf[indices])
        elapsed = timed(fn, n_iters=20, warmup_iters=3)
        bw = 2 * len(indices) * 4 / elapsed / 1e12
        results.append({"stride_bytes": stride_b, "bw_TBps": bw,
                        "tlb_pressure": "high" if stride_b >= 4096 else "low"})
    return results

# ───────────────────────────────────────────────────────────────────────────
# 5. GEMM Peak — Tensor Cores across dtypes
# ───────────────────────────────────────────────────────────────────────────
def bench_gemm(M=16384, N=16384, K=16384, dtype=torch.float16, n_iters=20):
    """Measure achieved TFLOPS for a large square GEMM."""
    a = torch.randn(M, K, device=DEVICE, dtype=dtype)
    b = torch.randn(K, N, device=DEVICE, dtype=dtype)
    def fn():
        torch.mm(a, b)
    elapsed = timed(fn, n_iters=n_iters, warmup_iters=5)
    flops = 2 * M * N * K
    tflops = flops / elapsed / 1e12
    return tflops

def bench_all_dtypes_gemm():
    results = {}
    configs = {
        "fp32":  torch.float32,
        "bf16":  torch.bfloat16,
        "fp16":  torch.float16,
        "fp8":   None,   # handled separately
        "int8":  None,
    }
    M = N = K = 8192
    for dtype_name, dtype in configs.items():
        if dtype is None:
            # FP8 via torch._scaled_mm if available, else skip
            if dtype_name == "fp8":
                try:
                    a = torch.randn(M, K, device=DEVICE).to(torch.float8_e4m3fn)
                    b = torch.randn(K, N, device=DEVICE).to(torch.float8_e4m3fn)
                    scale_a = torch.tensor(1.0, device=DEVICE)
                    scale_b = torch.tensor(1.0, device=DEVICE)
                    def fn_fp8():
                        torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                                         out_dtype=torch.bfloat16)
                    elapsed = timed(fn_fp8, n_iters=20, warmup_iters=5)
                    tflops = 2 * M * N * K / elapsed / 1e12
                    results["fp8"] = tflops
                except Exception as e:
                    print(f"  FP8 GEMM skipped: {e}")
            elif dtype_name == "int8":
                try:
                    a = torch.randint(-128, 127, (M, K), device=DEVICE, dtype=torch.int8)
                    b = torch.randint(-128, 127, (K, N), device=DEVICE, dtype=torch.int8)
                    def fn_i8():
                        torch._int_mm(a, b)
                    elapsed = timed(fn_i8, n_iters=20, warmup_iters=5)
                    tflops = 2 * M * N * K / elapsed / 1e12
                    results["int8"] = tflops
                except Exception as e:
                    print(f"  INT8 GEMM skipped: {e}")
        else:
            try:
                tflops = bench_gemm(M, N, K, dtype=dtype, n_iters=20)
                results[dtype_name] = tflops
            except Exception as e:
                print(f"  {dtype_name} GEMM skipped: {e}")
    return results

# ───────────────────────────────────────────────────────────────────────────
# 6. Max Power Stress (SM saturation)
# ───────────────────────────────────────────────────────────────────────────
def bench_max_power(duration_s: float = 10.0):
    """Run continuous large GEMMs and sample power."""
    M = N = K = 8192
    a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
    rec = TelemetryRecorder(gpu_indices=[0], hz=20)
    rec.start()
    t_end = time.time() + duration_s
    while time.time() < t_end:
        torch.mm(a, b)
    torch.cuda.synchronize()
    df = rec.stop()
    max_power  = df["power_w"].max()
    mean_power = df["power_w"].mean()
    max_temp   = df["temp_c"].max()
    return {"max_power_W": max_power, "mean_power_W": mean_power,
            "max_temp_C": max_temp, "df": df}

# ───────────────────────────────────────────────────────────────────────────
# 7. HBM Latency (pointer-chasing)
# ───────────────────────────────────────────────────────────────────────────
def bench_hbm_latency_approx():
    """
    Approximate latency: small random gather forces cache misses.
    Measure ns per element for tiny gather from large buffer.
    """
    results = []
    buf_sizes_mb = [1, 4, 16, 64, 256, 1024, 8192]
    for mb in buf_sizes_mb:
        n = (mb * 1024 * 1024) // 4
        buf = torch.randn(n, device=DEVICE, dtype=torch.float32)
        n_reads = min(1024, n)
        idx = torch.randint(0, n, (n_reads,), device=DEVICE)
        out = torch.empty(n_reads, device=DEVICE, dtype=torch.float32)
        def fn():
            out.copy_(buf[idx])
        elapsed = timed(fn, n_iters=100, warmup_iters=10)
        ns_per_elem = elapsed * 1e9 / n_reads
        results.append({"buf_size_mb": mb, "ns_per_read": ns_per_elem})
    return results

# ───────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  NVIDIA B200 System Profile — Phase 1")
    print("=" * 70)
    warmup()

    summary = {"gpu": "NVIDIA B200", "cuda_version": torch.version.cuda}
    summary.update(B200_SPECS)

    # 1. HBM Sequential BW
    print("\n[1/7] HBM Sequential Bandwidth...")
    bw_fp32 = bench_hbm_bw(size_gb=4.0, dtype=torch.float32)
    bw_fp16 = bench_hbm_bw(size_gb=4.0, dtype=torch.float16)
    print(f"  FP32 copy BW: {bw_fp32:.3f} TB/s  (theoretical: {MEM_BW_TBps} TB/s)")
    print(f"  FP16 copy BW: {bw_fp16:.3f} TB/s")
    bw_measured = max(bw_fp32, bw_fp16)
    summary["measured_hbm_seq_bw_TBps_fp32"] = bw_fp32
    summary["measured_hbm_seq_bw_TBps_fp16"] = bw_fp16
    summary["bw_utilization_pct"] = 100 * bw_measured / MEM_BW_TBps

    # 2. Random Access BW
    print("\n[2/7] Random Access Bandwidth...")
    bw_rand = bench_random_access(size_gb=0.5)
    print(f"  Random gather BW: {bw_rand:.4f} TB/s")
    summary["measured_hbm_rand_bw_TBps"] = bw_rand
    summary["rand_vs_seq_pct"] = 100 * bw_rand / bw_fp32

    # 3. Cache Hierarchy
    print("\n[3/7] Cache Hierarchy BW vs Working Set...")
    cache_data = bench_cache_hierarchy()
    df_cache = pd.DataFrame(cache_data)
    df_cache.to_csv(f"{OUT}/cache_hierarchy.csv", index=False)
    print(df_cache.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(df_cache["size_kb"], df_cache["bw_TBps"] * 1000, "o-", color="#1f77b4", lw=2)
    ax.axhline(MEM_BW_TBps * 1000, ls="--", color="red", label=f"Theoretical HBM {MEM_BW_TBps} TB/s")
    ax.set_xlabel("Working Set Size (KB, log scale)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("Cache Hierarchy Bandwidth — NVIDIA B200\n(L1 / L2 / HBM transitions)")
    ax.legend(); ax.grid(alpha=0.3)
    # Annotate approximate cache boundaries
    ax.axvspan(0, 256, alpha=0.08, color="green", label="L1 region")
    ax.axvspan(256, 50*1024, alpha=0.05, color="orange", label="L2 region")
    save_fig(fig, f"{GOUT}/cache_hierarchy_bw.png")

    # 4. TLB Stress
    print("\n[4/7] TLB Stress (Strided Access)...")
    tlb_data = bench_tlb()
    df_tlb = pd.DataFrame(tlb_data)
    df_tlb.to_csv(f"{OUT}/tlb_stress.csv", index=False)
    print(df_tlb.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(df_tlb["stride_bytes"], df_tlb["bw_TBps"] * 1000, "s-", color="#d62728", lw=2)
    ax.set_xlabel("Stride (bytes, log scale)"); ax.set_ylabel("BW (GB/s)")
    ax.set_title("TLB Stress — Effective BW vs Access Stride"); ax.grid(alpha=0.3)
    save_fig(fig, f"{GOUT}/tlb_stress.png")

    # 5. GEMM Throughput by dtype
    print("\n[5/7] GEMM Peak Throughput (Tensor Cores)...")
    gemm_results = bench_all_dtypes_gemm()
    summary["gemm_TFLOPS"] = gemm_results
    print("  Measured TFLOPS:")
    for k, v in gemm_results.items():
        pct = 100 * v / DTYPE_PEAK.get(k, 1)
        print(f"    {k:6s}: {v:8.1f} TFLOPS  ({pct:.1f}% of theoretical {DTYPE_PEAK.get(k,'?')} TFLOPS)")

    # Bar chart
    labels = list(gemm_results.keys())
    vals   = [gemm_results[k] for k in labels]
    theor  = [DTYPE_PEAK.get(k, 0) for k in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, theor, 0.4, label="Theoretical", color="#aec7e8")
    ax.bar(x + 0.2, vals,  0.4, label="Measured",    color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("TFLOPS"); ax.set_title("Peak GEMM Throughput — NVIDIA B200")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for xi, v in zip(x + 0.2, vals):
        ax.text(xi, v + 20, f"{v:.0f}", ha="center", fontsize=9)
    save_fig(fig, f"{GOUT}/gemm_peak_tflops.png")

    # 6. Max Power
    print("\n[6/7] Maximum Power (10 s GEMM stress)...")
    power_res = bench_max_power(duration_s=10)
    print(f"  Max power:  {power_res['max_power_W']:.1f} W  (TDP: {B200_SPECS['tdp_W']} W)")
    print(f"  Mean power: {power_res['mean_power_W']:.1f} W")
    print(f"  Max temp:   {power_res['max_temp_C']:.1f} °C")
    summary["max_power_W"]  = power_res["max_power_W"]
    summary["mean_power_W"] = power_res["mean_power_W"]
    summary["max_temp_C"]   = power_res["max_temp_C"]
    power_res["df"].to_csv(f"{OUT}/power_stress_timeseries.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df_pw = power_res["df"]
    t0 = df_pw["ts"].iloc[0]
    axes[0].plot(df_pw["ts"] - t0, df_pw["power_w"], color="#d62728", lw=1.5)
    axes[0].axhline(B200_SPECS["tdp_W"], ls="--", color="black", label="TDP 1000W")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Power (W)")
    axes[0].set_title("Power During GEMM Stress"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(df_pw["ts"] - t0, df_pw["temp_c"], color="#ff7f0e", lw=1.5)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Temp (°C)")
    axes[1].set_title("GPU Temperature During Stress"); axes[1].grid(alpha=0.3)
    save_fig(fig, f"{GOUT}/power_temp_stress.png")

    # 7. HBM Latency Approximation
    print("\n[7/7] HBM Latency Approximation...")
    lat_data = bench_hbm_latency_approx()
    df_lat = pd.DataFrame(lat_data)
    df_lat.to_csv(f"{OUT}/hbm_latency.csv", index=False)
    print(df_lat.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(df_lat["buf_size_mb"], df_lat["ns_per_read"], "D-", color="#2ca02c", lw=2)
    ax.set_xlabel("Buffer Size (MB, log)"); ax.set_ylabel("ns per Read")
    ax.set_title("Approximate Read Latency vs Buffer Size — B200"); ax.grid(alpha=0.3)
    save_fig(fig, f"{GOUT}/hbm_latency.png")
    summary["hbm_latency_ns_approx_4GB"] = df_lat.iloc[-1]["ns_per_read"]

    # ─── Roofline Plot ────────────────────────────────────────────────────
    print("\n  Generating Roofline plots...")
    roofline_points = {}
    bw = bw_measured
    for dtype_name, tflops in gemm_results.items():
        M = N = K = 8192
        dtype_sizes = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}
        es = dtype_sizes.get(dtype_name, 2)
        bytes_accessed = 2 * M * K * es + 2 * K * N * es  # A + B (rough)
        flops = 2 * M * N * K
        ai = flops / bytes_accessed
        roofline_points[dtype_name] = (ai, tflops)
    plot_roofline(roofline_points, dtype="bf16",
                  out_path=f"{GOUT}/roofline_system.png",
                  measured_bw_TBps=bw)

    # ─── Save JSON Summary ────────────────────────────────────────────────
    # Remove non-serializable df; convert numpy scalars to Python native
    def to_native(v):
        if isinstance(v, dict):
            return {kk: to_native(vv) for kk, vv in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    out_summary = {k: to_native(v) for k, v in summary.items()
                   if not isinstance(v, pd.DataFrame)}
    with open(f"{OUT}/system_profile_summary.json", "w") as f:
        json.dump(out_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  Phase 1 Complete. Results in results/system_profile/")
    print("=" * 70)
    print(json.dumps(out_summary, indent=2))

if __name__ == "__main__":
    main()
