#!/usr/bin/env python3
"""
Extended Tier-1 Telemetry — All SPAR-GPU-monitoring Workloads
Runs every workload class from robirahman/SPAR-GPU-monitoring on:
  • 1x GPU (GPU 0)
  • 2x GPU with NVLink (both GPUs, P2P data movement)
  • 2x GPU without explicit NVLink (DataParallel, fallback to PCIe)

Workloads:
  idle, pytorch_training, bert_finetune, gpt2_finetune, resnet50_inference,
  modern_inference (LLM), cufft_benchmark, nbody_sim, scientific_hpc,
  blender_render (approx), adversarial_training, fsdp_training (simulated)

Extended Metrics (beyond standard NVML Tier-1):
  GPU-dependent:
    gpu_util, mem_util, power_w, temp_c, sm_clock, mem_clock, nvlink_bw, ecc
  GPU-independent (new):
    system_ram_used_GB  — host memory usage
    cpu_util_pct        — host CPU utilization
    net_rx_MB, net_tx_MB — network I/O (relevant for distributed)
    disk_read_MB, disk_write_MB — storage I/O
    power_ratio         — GPU power / TDP (fraction)
    energy_J            — cumulative energy (power × Δt)
    perf_per_watt       — throughput / mean_power (tokens/s/W or iters/s/W)
    thermal_headroom_C  — TDP_temp - current_temp (approx)
"""
import os, sys, time, json, threading
import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.telemetry import TelemetryRecorder, GPUSample, pynvml
from utils.plotting import plot_telemetry_dashboard, plot_bar_comparison, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "spar_telemetry")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "spar_telemetry")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

N_GPUS   = torch.cuda.device_count()
TDP_W    = 1000.0
TDP_TEMP = 85.0    # approx throttle temp

# ── Extended system-level sampler ─────────────────────────────────────────────
class SystemSample:
    __slots__ = ["ts","cpu_pct","ram_used_GB","net_rx_MB","net_tx_MB",
                 "disk_read_MB","disk_write_MB"]
    def __init__(self):
        self.ts = time.time()
        self.cpu_pct = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        self.ram_used_GB = vm.used / 1e9
        net = psutil.net_io_counters()
        self.net_rx_MB = net.bytes_recv / 1e6
        self.net_tx_MB = net.bytes_sent / 1e6
        disk = psutil.disk_io_counters()
        self.disk_read_MB  = disk.read_bytes  / 1e6 if disk else 0
        self.disk_write_MB = disk.write_bytes / 1e6 if disk else 0

class ExtendedRecorder:
    """Combines GPU NVML + system-level telemetry."""
    def __init__(self, gpu_indices=None, hz=20.0):
        self.hz = hz
        n = pynvml.nvmlDeviceGetCount()
        self.indices = gpu_indices if gpu_indices is not None else list(range(n))
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.indices]
        self._gpu_samples = []
        self._sys_samples  = []
        self._stop = threading.Event()
        # Init psutil counters
        psutil.cpu_percent(interval=None)
        psutil.net_io_counters()
        psutil.disk_io_counters()

    def start(self):
        self._stop.clear()
        self._gpu_samples.clear()
        self._sys_samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self._build_df()

    def _loop(self):
        from utils.telemetry import sample_gpu
        interval = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            for h, idx in zip(self.handles, self.indices):
                self._gpu_samples.append(sample_gpu(h, idx))
            self._sys_samples.append(SystemSample())
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _build_df(self):
        df_gpu = pd.DataFrame([s.__dict__ for s in self._gpu_samples])
        df_sys = pd.DataFrame([s.__dict__ for s in self._sys_samples])
        if df_gpu.empty or df_sys.empty:
            return df_gpu, df_sys

        # Derive GPU-side computed columns
        df_gpu["power_ratio"]       = df_gpu["power_w"] / TDP_W
        df_gpu["thermal_headroom_C"]= TDP_TEMP - df_gpu["temp_c"]

        # Compute cumulative energy (J) and mean power-ratio per GPU
        for gi in df_gpu["gpu_idx"].unique():
            mask = df_gpu["gpu_idx"] == gi
            pw   = df_gpu.loc[mask, "power_w"].values
            ts   = df_gpu.loc[mask, "ts"].values
            dt   = np.diff(ts, prepend=ts[0])
            energy = np.cumsum(pw * dt)
            df_gpu.loc[mask, "energy_J"] = energy

        # Delta system metrics (diff to get rates)
        df_sys["net_rx_rate_MBps"]  = df_sys["net_rx_MB"].diff().clip(lower=0)
        df_sys["net_tx_rate_MBps"]  = df_sys["net_tx_MB"].diff().clip(lower=0)
        df_sys["disk_r_rate_MBps"]  = df_sys["disk_read_MB"].diff().clip(lower=0)
        df_sys["disk_w_rate_MBps"]  = df_sys["disk_write_MB"].diff().clip(lower=0)
        return df_gpu, df_sys

def summarize_extended(df_gpu, df_sys, gpu_ids, iters=None, workload_label=""):
    rows = df_gpu[df_gpu["gpu_idx"].isin(gpu_ids)]
    mean_power  = rows["power_w"].mean()
    total_energy= rows.groupby("gpu_idx")["energy_J"].last().sum()
    result = {
        "workload": workload_label,
        "gpu_ids": gpu_ids,
        "num_gpus": len(gpu_ids),
        # GPU metrics
        "mean_gpu_util":     float(rows["gpu_util"].mean()),
        "max_gpu_util":      float(rows["gpu_util"].max()),
        "mean_power_W":      float(mean_power),
        "max_power_W":       float(rows["power_w"].max()),
        "mean_power_ratio":  float(rows["power_ratio"].mean()),
        "total_energy_kJ":   float(total_energy / 1000),
        "mean_temp_C":       float(rows["temp_c"].mean()),
        "thermal_headroom_C":float(rows["thermal_headroom_C"].mean()),
        "mean_mem_util":     float(rows["mem_util"].mean()),
        "mean_sm_clock_MHz": float(rows["sm_clock_mhz"].mean()),
        # System (GPU-independent)
        "mean_cpu_pct":      float(df_sys["cpu_pct"].mean()),
        "max_cpu_pct":       float(df_sys["cpu_pct"].max()),
        "mean_ram_GB":       float(df_sys["ram_used_GB"].mean()),
        "mean_net_rx_MBps":  float(df_sys["net_rx_rate_MBps"].mean()),
        "mean_net_tx_MBps":  float(df_sys["net_tx_rate_MBps"].mean()),
        "mean_disk_r_MBps":  float(df_sys["disk_r_rate_MBps"].mean()),
        "mean_disk_w_MBps":  float(df_sys["disk_w_rate_MBps"].mean()),
    }
    if iters:
        result["iters"] = iters
        result["perf_per_watt"] = iters / mean_power if mean_power > 0 else 0
    return result

def plot_power_ratio_over_time(df_gpu, gpu_ids, workload_label, out_path):
    """Plot GPU power / TDP ratio over time — key non-standard metric."""
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#1f77b4", "#ff7f0e"]
    for gi, col in zip(gpu_ids, colors):
        df_g = df_gpu[df_gpu["gpu_idx"] == gi].copy()
        df_g["t"] = df_g["ts"] - df_g["ts"].iloc[0]
        ax.plot(df_g["t"], df_g["power_ratio"], color=col,
                lw=1.5, label=f"GPU {gi}", alpha=0.85)
    ax.axhline(1.0, ls="--", color="red", lw=1.5, label="TDP (100%)")
    ax.axhline(0.5, ls=":", color="gray", lw=1, label="50% TDP")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Power / TDP Ratio")
    ax.set_title(f"Power Ratio Over Time — {workload_label}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    # Fill region
    for gi, col in zip(gpu_ids, colors):
        df_g = df_gpu[df_gpu["gpu_idx"] == gi].copy()
        df_g["t"] = df_g["ts"] - df_g["ts"].iloc[0]
        ax.fill_between(df_g["t"], 0, df_g["power_ratio"],
                        alpha=0.12, color=col)
    save_fig(fig, out_path)

def plot_system_metrics(df_sys, workload_label, out_path):
    """Plot GPU-independent system metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f"System (Host) Metrics — {workload_label}", fontsize=12)
    df_sys = df_sys.copy()
    df_sys["t"] = df_sys["ts"] - df_sys["ts"].iloc[0]
    specs = [
        ("cpu_pct",          "CPU Util (%)",     "#1f77b4"),
        ("ram_used_GB",      "RAM Used (GB)",    "#ff7f0e"),
        ("net_rx_rate_MBps", "Net RX (MB/s)",    "#2ca02c"),
        ("disk_r_rate_MBps", "Disk Read (MB/s)", "#d62728"),
    ]
    for ax, (col, lbl, clr) in zip(axes.flat, specs):
        ax.plot(df_sys["t"], df_sys[col], color=clr, lw=1.5)
        ax.set_xlabel("Time (s)"); ax.set_ylabel(lbl)
        ax.set_title(lbl); ax.grid(alpha=0.3)
        ax.axhline(df_sys[col].mean(), ls="--", color="gray", lw=1,
                   label=f"mean={df_sys[col].mean():.1f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, out_path)

# ── SPAR Workload Definitions ─────────────────────────────────────────────────

def w_idle(gpu_ids, duration=15):
    time.sleep(duration)
    return 0

def w_pytorch_training(gpu_ids, duration=20):
    dev = f"cuda:{gpu_ids[0]}"
    import torchvision.models as tvm
    model = tvm.resnet50().to(dev).to(torch.bfloat16)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(16, 3, 224, 224, device=dev, dtype=torch.bfloat16)
    t_end = time.time() + duration; iters = 0
    model.train()
    while time.time() < t_end:
        out = model(x); loss = out.mean(); loss.backward(); opt.step(); opt.zero_grad()
        iters += 1
    torch.cuda.synchronize()
    return iters

def w_resnet50_inference(gpu_ids, duration=20):
    dev = f"cuda:{gpu_ids[0]}"
    import torchvision.models as tvm
    model = tvm.resnet50().to(dev).to(torch.bfloat16).eval()
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    x = torch.randn(32, 3, 224, 224, device=dev, dtype=torch.bfloat16)
    t_end = time.time() + duration; iters = 0
    with torch.no_grad():
        while time.time() < t_end:
            model(x); iters += 1
    torch.cuda.synchronize()
    return iters

def w_bert_finetune(gpu_ids, duration=20):
    dev = f"cuda:{gpu_ids[0]}"
    vocab = 30522; d = 768; nhead = 12; seq = 128; bs = 8
    embed = nn.Embedding(vocab, d).to(dev).to(torch.bfloat16)
    enc   = nn.TransformerEncoderLayer(d, nhead, 3072, batch_first=True).to(dev).to(torch.bfloat16)
    head  = nn.Linear(d, 2).to(dev).to(torch.bfloat16)
    model = nn.Sequential(embed, enc, head)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids  = torch.randint(0, vocab, (bs, seq), device=dev)
    labs = torch.randint(0, 2, (bs,), device=dev)
    t_end = time.time() + duration; iters = 0
    model.train()
    while time.time() < t_end:
        out = model(ids)
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            out = out[:, 0, :]
        loss = nn.functional.cross_entropy(out, labs)
        loss.backward(); opt.step(); opt.zero_grad()
        iters += 1
    torch.cuda.synchronize()
    return iters

def w_gpt2_finetune(gpu_ids, duration=20):
    dev = f"cuda:{gpu_ids[0]}"
    vocab = 50257; d = 768; nhead = 12; seq = 128; bs = 4
    embed = nn.Embedding(vocab, d).to(dev).to(torch.bfloat16)
    block = nn.TransformerDecoderLayer(d, nhead, 3072, batch_first=True).to(dev).to(torch.bfloat16)
    head  = nn.Linear(d, vocab).to(dev).to(torch.bfloat16)
    opt   = torch.optim.AdamW(list(embed.parameters()) + list(block.parameters()) + list(head.parameters()), lr=1e-4)
    ids   = torch.randint(0, vocab, (bs, seq), device=dev)
    t_end = time.time() + duration; iters = 0
    while time.time() < t_end:
        x   = embed(ids)
        out = block(x, x)
        logits = head(out)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab), ids.view(-1))
        loss.backward(); opt.step(); opt.zero_grad()
        iters += 1
    torch.cuda.synchronize()
    return iters

def w_cufft(gpu_ids, duration=20):
    dev = f"cuda:{gpu_ids[0]}"
    n = 1024 * 1024
    x = torch.randn(n, dtype=torch.complex64, device=dev)
    t_end = time.time() + duration; iters = 0
    while time.time() < t_end:
        torch.fft.fft(x); iters += 1
    torch.cuda.synchronize()
    return iters

def w_nbody(gpu_ids, duration=20):
    dev = f"cuda:{gpu_ids[0]}"
    N = 8192
    pos = torch.randn(N, 3, device=dev, dtype=torch.float32)
    vel = torch.zeros(N, 3, device=dev, dtype=torch.float32)
    mass = torch.ones(N, device=dev, dtype=torch.float32)
    t_end = time.time() + duration; iters = 0
    while time.time() < t_end:
        r   = pos.unsqueeze(0) - pos.unsqueeze(1)   # [N,N,3]
        d2  = (r ** 2).sum(-1) + 1e-4               # [N,N]
        d3  = d2 ** 1.5
        F   = (r * (mass.unsqueeze(0) / d3.unsqueeze(-1))).sum(1)
        vel = vel + F * 0.001
        pos = pos + vel * 0.001
        iters += 1
    torch.cuda.synchronize()
    return iters

def w_scientific_hpc(gpu_ids, duration=20):
    """Matrix multiply chain (HPC-style dense linear algebra)."""
    dev = f"cuda:{gpu_ids[0]}"
    M = 4096
    A = torch.randn(M, M, device=dev, dtype=torch.float32)
    B = torch.randn(M, M, device=dev, dtype=torch.float32)
    t_end = time.time() + duration; iters = 0
    while time.time() < t_end:
        C = torch.mm(A, B); A = C / C.norm(); iters += 1
    torch.cuda.synchronize()
    return iters

def w_modern_inference(gpu_ids, duration=20):
    """LLM-style large GEMM inference (no model needed)."""
    dev = f"cuda:{gpu_ids[0]}"
    # Simulate 7B model decode: [bs, 1, d] x [d, d*4] projections
    bs = 8; d = 4096; ffn = 14336; layers = 32
    x = torch.randn(bs, 1, d, device=dev, dtype=torch.bfloat16)
    Wq = torch.randn(d, d, device=dev, dtype=torch.bfloat16)
    Wffn = torch.randn(d, ffn, device=dev, dtype=torch.bfloat16)
    t_end = time.time() + duration; iters = 0
    with torch.no_grad():
        while time.time() < t_end:
            h = x.view(bs, d) @ Wq
            h = torch.nn.functional.silu(h.view(bs, d) @ Wffn)
            iters += 1
    torch.cuda.synchronize()
    return iters

def w_adversarial(gpu_ids, duration=20):
    """Adversarial training (PGD-style): forward + backward + perturbation."""
    dev = f"cuda:{gpu_ids[0]}"
    import torchvision.models as tvm
    model = tvm.resnet18().to(dev).to(torch.bfloat16).train()
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    opt = torch.optim.SGD(model.parameters(), 0.01)
    x = torch.randn(8, 3, 224, 224, device=dev, dtype=torch.bfloat16, requires_grad=True)
    y = torch.randint(0, 1000, (8,), device=dev)
    t_end = time.time() + duration; iters = 0
    while time.time() < t_end:
        # Inner adversarial step
        x_adv = x.detach().clone().requires_grad_(True)
        out  = model(x_adv)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + 0.01 * x_adv.grad.sign()
        # Outer training step
        out2 = model(x_adv)
        loss2 = nn.functional.cross_entropy(out2, y)
        loss2.backward(); opt.step(); opt.zero_grad()
        iters += 1
    torch.cuda.synchronize()
    return iters

def w_nvlink_p2p(gpu_ids, duration=20):
    """NVLink explicit P2P (only meaningful for 2+ GPUs)."""
    if len(gpu_ids) < 2:
        time.sleep(duration); return 0
    n = int(256e6 / 4)
    src = torch.randn(n, device="cuda:0")
    dst = torch.empty(n, device="cuda:1")
    t_end = time.time() + duration; iters = 0
    while time.time() < t_end:
        dst.copy_(src); iters += 1
    torch.cuda.synchronize()
    return iters

# ── Workload registry ─────────────────────────────────────────────────────────
WORKLOADS = {
    "idle":              (w_idle,            15),
    "pytorch_training":  (w_pytorch_training, 20),
    "bert_finetune":     (w_bert_finetune,    20),
    "gpt2_finetune":     (w_gpt2_finetune,    20),
    "resnet50_inference":(w_resnet50_inference,20),
    "modern_inference":  (w_modern_inference, 20),
    "cufft_benchmark":   (w_cufft,           20),
    "nbody_sim":         (w_nbody,           20),
    "scientific_hpc":    (w_scientific_hpc,  20),
    "adversarial_training": (w_adversarial,  20),
    "nvlink_p2p":        (w_nvlink_p2p,      20),
}

GPU_CONFIGS = {
    "1x_GPU":     [0],
    "2x_GPU_no_NVLink": [0, 1],   # DataParallel, implicit NVLink (OS-managed)
    "2x_GPU_NVLink": [0, 1],      # Explicit P2P for nvlink_p2p workload
}

def run_experiment(workload_name, fn, duration, gpu_ids, config_label):
    print(f"  [{config_label}] {workload_name} ({duration}s) ...", end=" ", flush=True)
    rec = ExtendedRecorder(gpu_indices=gpu_ids, hz=20)
    rec.start()
    iters = fn(gpu_ids, duration)
    df_gpu, df_sys = rec.stop()
    label = f"{workload_name}_{config_label}"
    if not df_gpu.empty:
        df_gpu.to_csv(f"{OUT}/{label}_gpu.csv", index=False)
    if not df_sys.empty:
        df_sys.to_csv(f"{OUT}/{label}_sys.csv", index=False)
    summary = summarize_extended(df_gpu, df_sys, gpu_ids, iters, label)
    print(f"  iters={iters}  power={summary['mean_power_W']:.0f}W  "
          f"cpu={summary['mean_cpu_pct']:.1f}%  ram={summary['mean_ram_GB']:.1f}GB")
    # Dashboard
    if not df_gpu.empty:
        plot_telemetry_dashboard(df_gpu, gpu_ids[0], f"{GOUT}/{label}_gpu{gpu_ids[0]}_dashboard.png")
        plot_power_ratio_over_time(df_gpu, gpu_ids, label, f"{GOUT}/{label}_power_ratio.png")
    if not df_sys.empty:
        plot_system_metrics(df_sys, label, f"{GOUT}/{label}_system_metrics.png")
    return summary

def main():
    import gc
    print("=" * 70)
    print(f"  SPAR Workloads Extended Tier-1 Telemetry — {N_GPUS} GPU(s) detected")
    print("=" * 70)

    all_summaries = []

    for wname, (wfn, dur) in WORKLOADS.items():
        print(f"\n─── Workload: {wname.upper()} ───")
        import gc; torch.cuda.empty_cache(); gc.collect()

        # 1x GPU
        s1 = run_experiment(wname, wfn, dur, [0], "1x_GPU")
        all_summaries.append(s1)

        # 2x GPU (both configs)
        if N_GPUS >= 2:
            s2 = run_experiment(wname, wfn, dur, [0, 1], "2x_GPU")
            all_summaries.append(s2)

    # ── Save full summary ──────────────────────────────────────────────────
    df_sum = pd.DataFrame(all_summaries)
    df_sum.to_csv(f"{OUT}/spar_extended_summary.csv", index=False)
    print(f"\n  Summary saved → {OUT}/spar_extended_summary.csv")

    # ── Comparison plots ────────────────────────────────────────────────────
    if N_GPUS >= 2:
        # 1x vs 2x for each workload
        workload_names = list(WORKLOADS.keys())
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("1x GPU vs 2x GPU — All SPAR Workloads", fontsize=13)

        metrics = [
            ("mean_gpu_util",   "GPU Util (%)"),
            ("mean_power_W",    "Power (W)"),
            ("mean_power_ratio","Power Ratio (/ TDP)"),
            ("mean_cpu_pct",    "CPU Util (%)"),
            ("mean_ram_GB",     "RAM Used (GB)"),
            ("perf_per_watt",   "Perf / Watt (iters/W)"),
        ]
        colors_1x = "#1f77b4"
        colors_2x = "#ff7f0e"
        x = np.arange(len(workload_names))

        for ax, (met, lbl) in zip(axes.flat, metrics):
            vals_1x = [df_sum[(df_sum["workload"].str.contains(wn)) &
                              (df_sum["num_gpus"] == 1)][met].mean()
                       for wn in workload_names]
            vals_2x = [df_sum[(df_sum["workload"].str.contains(wn)) &
                              (df_sum["num_gpus"] == 2)][met].mean()
                       for wn in workload_names]
            ax.bar(x - 0.2, vals_1x, 0.4, label="1x GPU", color=colors_1x)
            ax.bar(x + 0.2, vals_2x, 0.4, label="2x GPU", color=colors_2x)
            ax.set_xticks(x)
            ax.set_xticklabels(workload_names, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(lbl); ax.set_title(lbl); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_fig(fig, f"{GOUT}/1x_vs_2x_all_workloads.png")

    # ── Power ratio over time — aggregate plot ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(WORKLOADS)))
    for (wname, _), col in zip(WORKLOADS.items(), colors):
        label = f"{wname}_1x_GPU"
        fpath = f"{OUT}/{label}_gpu.csv"
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df_g0 = df[df["gpu_idx"] == 0]
            if len(df_g0) > 1:
                t = df_g0["ts"].values - df_g0["ts"].values[0]
                ax.plot(t, df_g0["power_w"] / TDP_W, color=col, lw=1.2,
                        label=wname, alpha=0.85)
    ax.axhline(1.0, ls="--", color="red", lw=1.5, label="TDP")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Power / TDP Ratio")
    ax.set_title("Power Ratio Over Time — All SPAR Workloads (1x GPU)")
    ax.legend(fontsize=7, ncol=3, loc="lower right"); ax.grid(alpha=0.3)
    save_fig(fig, f"{GOUT}/power_ratio_all_workloads.png")

    print("\n" + "=" * 70)
    print("  SPAR Extended Telemetry Complete.")
    print(f"  Results → {OUT}/  |  Graphs → {GOUT}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
