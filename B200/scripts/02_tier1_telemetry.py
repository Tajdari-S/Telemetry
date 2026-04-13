#!/usr/bin/env python3
"""
Phase 4 — Tier-1 NVML Telemetry Tests
Mirrors the SPAR-GPU-monitoring tier-1 collection methodology.
Runs workloads on:
  • 1x GPU (GPU 0 only)
  • 2x GPU (DataParallel / both GPUs)
  • NVLink P2P transfer stress
Collects pynvml telemetry at 20 Hz throughout.
"""
import os, sys, time, json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.telemetry import TelemetryRecorder
from utils.plotting import plot_telemetry_dashboard, plot_bar_comparison, plot_time_series, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "telemetry")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "telemetry")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

N_GPUS = torch.cuda.device_count()
print(f"Detected {N_GPUS} GPU(s)")

# ───────────────────────────────────────────────────────────────────────────
# Workload: Continuous GEMM (simple, reliable stress)
# ───────────────────────────────────────────────────────────────────────────
def run_gemm_workload(device_ids, duration_s=20, M=8192, dtype=torch.bfloat16):
    """Run GEMM on specified devices for `duration_s` seconds."""
    if len(device_ids) == 1:
        dev = torch.device(f"cuda:{device_ids[0]}")
        a = torch.randn(M, M, device=dev, dtype=dtype)
        b = torch.randn(M, M, device=dev, dtype=dtype)
        t_end = time.time() + duration_s
        iters = 0
        while time.time() < t_end:
            torch.mm(a, b)
            iters += 1
        torch.cuda.synchronize()
        return iters
    else:
        # Both GPUs independently
        import threading
        results = {}
        def worker(gpu_id):
            dev = torch.device(f"cuda:{gpu_id}")
            a = torch.randn(M, M, device=dev, dtype=dtype)
            b = torch.randn(M, M, device=dev, dtype=dtype)
            t_end = time.time() + duration_s
            iters = 0
            while time.time() < t_end:
                torch.mm(a, b)
                iters += 1
            torch.cuda.synchronize(dev)
            results[gpu_id] = iters
        threads = [threading.Thread(target=worker, args=(g,)) for g in device_ids]
        for t in threads: t.start()
        for t in threads: t.join()
        return sum(results.values())

# ───────────────────────────────────────────────────────────────────────────
# Workload: NVLink P2P stress
# ───────────────────────────────────────────────────────────────────────────
def run_nvlink_stress(duration_s=20, tensor_gb=1.0):
    """Continuously copy tensors GPU0→GPU1 via NVLink."""
    n = int(tensor_gb * 1e9 / 4)
    src = torch.randn(n, device="cuda:0", dtype=torch.float32)
    dst = torch.empty(n, device="cuda:1", dtype=torch.float32)
    t_end = time.time() + duration_s
    iters = 0
    while time.time() < t_end:
        dst.copy_(src)
        iters += 1
    torch.cuda.synchronize()
    bytes_transferred = iters * tensor_gb * 1e9
    bw_GBps = bytes_transferred / duration_s / 1e9
    return iters, bw_GBps

# ───────────────────────────────────────────────────────────────────────────
# Workload: Transformer-like (attention + FFN on GPU)
# ───────────────────────────────────────────────────────────────────────────
class MiniTransformerBlock(nn.Module):
    def __init__(self, d=4096, nhead=32, ffn_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Linear(d * ffn_mult, d)
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        ax, _ = self.attn(x, x, x)
        x = self.ln1(x + ax)
        x = self.ln2(x + self.ffn(x))
        return x

def run_transformer_workload(device_ids, duration_s=20, batch=4, seq=512, d=4096):
    dev = torch.device(f"cuda:{device_ids[0]}")
    model = MiniTransformerBlock(d=d).to(dev).to(torch.bfloat16)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    x = torch.randn(batch, seq, d, device=dev, dtype=torch.bfloat16)
    t_end = time.time() + duration_s
    iters = 0
    with torch.no_grad():
        while time.time() < t_end:
            model(x)
            iters += 1
    torch.cuda.synchronize()
    return iters

# ───────────────────────────────────────────────────────────────────────────
# Tier-1 Telemetry: collect and compare
# ───────────────────────────────────────────────────────────────────────────
def run_tier1_experiment(name: str, workload_fn, device_ids, duration_s=20):
    """Run a workload and collect telemetry. Returns (iters, df)."""
    print(f"  Running '{name}' on GPU(s) {device_ids} for {duration_s}s...")
    rec = TelemetryRecorder(gpu_indices=device_ids, hz=20)
    rec.start()
    result = workload_fn()
    df = rec.stop()
    df.to_csv(f"{OUT}/telemetry_{name.replace(' ', '_')}.csv", index=False)
    print(f"    Done. {len(df)} samples, {result} iters")
    return result, df

def summarize(df: pd.DataFrame, device_ids) -> dict:
    rows = df[df["gpu_idx"].isin(device_ids)]
    return {
        "mean_gpu_util":   rows["gpu_util"].mean(),
        "max_gpu_util":    rows["gpu_util"].max(),
        "mean_power_W":    rows["power_w"].mean(),
        "max_power_W":     rows["power_w"].max(),
        "mean_temp_C":     rows["temp_c"].mean(),
        "mean_mem_util":   rows["mem_util"].mean(),
        "mean_sm_clock":   rows["sm_clock_mhz"].mean(),
        "mean_mem_clock":  rows["mem_clock_mhz"].mean(),
    }

def main():
    print("=" * 70)
    print("  Tier-1 NVML Telemetry Tests — NVIDIA B200")
    print("=" * 70)

    results_summary = {}
    DURATION = 25  # seconds per experiment

    # ── 1x GPU GEMM ─────────────────────────────────────────────────────────
    print("\n[1/6] 1x GPU GEMM stress (GPU 0)...")
    iters, df = run_tier1_experiment(
        "1xGPU_GEMM", lambda: run_gemm_workload([0], DURATION), [0], DURATION)
    results_summary["1xGPU_GEMM"] = summarize(df, [0])
    results_summary["1xGPU_GEMM"]["iters"] = iters
    plot_telemetry_dashboard(df, 0, f"{GOUT}/dashboard_1xgpu_gemm_gpu0.png")

    # ── 2x GPU GEMM ─────────────────────────────────────────────────────────
    if N_GPUS >= 2:
        print("\n[2/6] 2x GPU GEMM stress (both GPUs)...")
        iters, df = run_tier1_experiment(
            "2xGPU_GEMM", lambda: run_gemm_workload([0, 1], DURATION), [0, 1], DURATION)
        results_summary["2xGPU_GEMM"] = summarize(df, [0, 1])
        results_summary["2xGPU_GEMM"]["iters"] = iters
        for gi in [0, 1]:
            plot_telemetry_dashboard(df, gi, f"{GOUT}/dashboard_2xgpu_gemm_gpu{gi}.png")
    else:
        print("  Skipping 2x GPU (only 1 GPU available)")

    # ── NVLink P2P Stress ────────────────────────────────────────────────────
    if N_GPUS >= 2:
        print("\n[3/6] NVLink P2P stress (GPU0→GPU1)...")
        iters, df = run_tier1_experiment(
            "NVLink_P2P", lambda: run_nvlink_stress(DURATION, tensor_gb=1.0)[0],
            [0, 1], DURATION)
        _, bw = run_nvlink_stress(10, tensor_gb=1.0)
        results_summary["NVLink_P2P"] = summarize(df, [0, 1])
        results_summary["NVLink_P2P"]["iters"] = iters
        results_summary["NVLink_P2P"]["measured_bw_GBps"] = bw
        print(f"    NVLink P2P BW: {bw:.1f} GB/s (theoretical: 900 GB/s)")
        plot_telemetry_dashboard(df, 0, f"{GOUT}/dashboard_nvlink_gpu0.png")
        plot_telemetry_dashboard(df, 1, f"{GOUT}/dashboard_nvlink_gpu1.png")
    else:
        print("  Skipping NVLink (only 1 GPU available)")

    # ── 1x GPU Transformer ───────────────────────────────────────────────────
    print("\n[4/6] 1x GPU Transformer block workload (GPU 0)...")
    iters, df = run_tier1_experiment(
        "1xGPU_Transformer", lambda: run_transformer_workload([0], DURATION),
        [0], DURATION)
    results_summary["1xGPU_Transformer"] = summarize(df, [0])
    results_summary["1xGPU_Transformer"]["iters"] = iters
    plot_telemetry_dashboard(df, 0, f"{GOUT}/dashboard_1xgpu_transformer.png")

    # ── 2x GPU Transformer (DataParallel) ───────────────────────────────────
    if N_GPUS >= 2:
        print("\n[5/6] 2x GPU Transformer block workload (DataParallel)...")
        iters, df = run_tier1_experiment(
            "2xGPU_Transformer", lambda: run_transformer_workload([0, 1], DURATION),
            [0, 1], DURATION)
        results_summary["2xGPU_Transformer"] = summarize(df, [0, 1])
        results_summary["2xGPU_Transformer"]["iters"] = iters
        for gi in [0, 1]:
            plot_telemetry_dashboard(df, gi, f"{GOUT}/dashboard_2xgpu_transformer_gpu{gi}.png")

    # ── Idle Baseline ────────────────────────────────────────────────────────
    print("\n[6/6] Idle baseline (15s)...")
    iters, df = run_tier1_experiment(
        "Idle", lambda: time.sleep(15) or 0, [0] + ([1] if N_GPUS >= 2 else []),
        15)
    results_summary["Idle"] = summarize(df, [0] + ([1] if N_GPUS >= 2 else []))
    plot_telemetry_dashboard(df, 0, f"{GOUT}/dashboard_idle_gpu0.png")

    # ── Comparison Bar Charts ────────────────────────────────────────────────
    print("\n  Generating comparison plots...")
    experiment_names = list(results_summary.keys())

    for metric, ylabel, title_sfx in [
        ("mean_power_W",  "Mean Power (W)",    "Mean Power"),
        ("max_gpu_util",  "Max GPU Util (%)",  "Peak GPU Utilization"),
        ("mean_gpu_util", "Mean GPU Util (%)", "Mean GPU Utilization"),
        ("mean_temp_C",   "Mean Temp (°C)",    "Mean Temperature"),
    ]:
        vals = [results_summary[n].get(metric, 0) for n in experiment_names]
        plot_bar_comparison(
            experiment_names, vals, ylabel,
            f"Tier-1 Telemetry Comparison — {title_sfx}",
            f"{GOUT}/compare_{metric}.png"
        )

    # NVLink throughput comparison
    if "NVLink_P2P" in results_summary and "measured_bw_GBps" in results_summary["NVLink_P2P"]:
        print(f"\n  NVLink P2P Bandwidth: {results_summary['NVLink_P2P']['measured_bw_GBps']:.1f} GB/s")

    # ── 1x vs 2x power comparison ────────────────────────────────────────────
    if N_GPUS >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metrics = ["mean_gpu_util", "mean_power_W"]
        labels  = ["GPU Util (%)", "Power (W)"]
        for ax, m, lbl in zip(axes, metrics, labels):
            pairs = {k: results_summary[k][m]
                     for k in results_summary if k != "Idle"}
            ax.bar(range(len(pairs)), list(pairs.values()),
                   color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(pairs)])
            ax.set_xticks(range(len(pairs)))
            ax.set_xticklabels(list(pairs.keys()), rotation=25, ha="right", fontsize=9)
            ax.set_ylabel(lbl); ax.set_title(f"{lbl} by Experiment")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("1x GPU vs 2x GPU vs NVLink — Tier-1 Telemetry", fontsize=13)
        plt.tight_layout()
        save_fig(fig, f"{GOUT}/1x_vs_2x_vs_nvlink_comparison.png")

    # ── Save Summary JSON ─────────────────────────────────────────────────────
    with open(f"{OUT}/tier1_telemetry_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  Tier-1 Telemetry Tests Complete. Results in results/telemetry/")
    print("=" * 70)
    print(json.dumps(results_summary, indent=2))

if __name__ == "__main__":
    main()
