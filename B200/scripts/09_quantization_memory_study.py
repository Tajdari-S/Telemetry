#!/usr/bin/env python3
"""
Study: Memory vs Compute effects of quantization — NVIDIA B200
Isolates three execution paths for a linear layer:
  1. BF16          — baseline: load 2B/param, matmul in BF16 TC
  2. INT8 native   — load 1B/param, matmul in INT8 TC (torch._int_mm), dequant output only
  3. INT8 dequant  — load 1B/param, dequant weights to BF16, matmul in BF16 TC (bitsandbytes path)
  4. FP8 native    — load 1B/param, matmul in FP8 TC (torch._scaled_mm)
  5. Dequant-only  — measures dequantization kernel alone (INT8→BF16 cast)

Also plots roofline positions for each path.
"""
import os, sys, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils.roofline import plot_roofline, MEM_BW_TBps, DTYPE_PEAK
from utils.plotting import save_fig

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "quant_study")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "quant_study")
os.makedirs(OUT,  exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

DEVICE  = "cuda"
TDP_W   = 1000.0
WARMUP  = 20
ITERS   = 200

# ── Sizes to sweep (M=batch*seq tokens, N=d_model, K=ffn_dim for a linear layer) ──
# Use shapes representative of a 7-8B LLM FFN projection
SHAPES = [
    (1,    4096, 14336),   # BS=1  decode  — memory-bound extreme
    (8,    4096, 14336),   # BS=8  decode
    (32,   4096, 14336),   # BS=32 decode  — approaching ridge
    (128,  4096, 14336),   # BS=128 prefill
    (512,  4096, 14336),   # BS=512 prefill — compute-bound territory
    (2048, 4096, 14336),   # Large prefill
]

# ── Benchmark helpers ──────────────────────────────────────────────────────────
def bench(fn, warmup=WARMUP, iters=ITERS):
    """Returns mean latency (ms) and throughput FLOP/s."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters   # ms per call

def tflops_from_gemm(M, N, K, ms):
    """TFLOPS for a M×K @ K×N matmul."""
    return 2 * M * N * K / (ms * 1e-3) / 1e12

def mem_bytes_gemm(M, N, K, a_bytes, b_bytes, c_bytes=2):
    """Approximate memory traffic for a matmul (load A, load B, write C)."""
    return M * K * a_bytes + K * N * b_bytes + M * N * c_bytes

# ── Execution paths ───────────────────────────────────────────────────────────
def run_bf16(M, N, K):
    W = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    ms = bench(lambda: torch.nn.functional.linear(x, W))
    tflops = tflops_from_gemm(M, N, K, ms)
    # Mem: load x (BF16) + load W (BF16) + write y (BF16)
    mem = mem_bytes_gemm(M, N, K, 2, 2, 2)
    ai = 2 * M * N * K / mem
    return ms, tflops, ai, mem

def run_fp8_native(M, N, K):
    """FP8 native via torch._scaled_mm — no dequantization.
    cuBLASLt requires: A row-major (M,K), B col-major (K,N).
    W stored as (N,K) row-major; W.T gives (K,N) col-major without copying.
    """
    W = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    x = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device=DEVICE)
    scale_b = torch.tensor(1.0, device=DEVICE)
    # W.T is (K,N) col-major — the layout cuBLASLt FP8 expects for B
    W_t = W.T   # do NOT call .contiguous() — that would make it row-major again
    try:
        ms = bench(lambda: torch._scaled_mm(x, W_t,
                                             scale_a=scale_a, scale_b=scale_b,
                                             out_dtype=torch.bfloat16))
        tflops = tflops_from_gemm(M, N, K, ms)
        mem = mem_bytes_gemm(M, N, K, 1, 1, 2)
        ai = 2 * M * N * K / mem
        return ms, tflops, ai, mem
    except Exception as e:
        print(f"    FP8 native skip: {e}")
        return None, None, None, None

def run_int8_native(M, N, K):
    """INT8 native via torch._int_mm — no dequantization of weights.
    Weights and activations in INT8, output accumulates in INT32.
    This is the 'pure memory bandwidth' path: loads 1 byte/param, no extra passes.
    torch._int_mm requires M >= 16; pad if needed.
    """
    M_actual = M
    M = max(M, 17)   # _int_mm requires M > 16 (strictly greater)
    W_fp = torch.randn(N, K, device=DEVICE)
    x_fp = torch.randn(M, K, device=DEVICE)
    W = (W_fp * 127 / W_fp.abs().max()).round().clamp(-128, 127).to(torch.int8)
    x = (x_fp * 127 / x_fp.abs().max()).round().clamp(-128, 127).to(torch.int8)
    # _int_mm requires contiguous row-major layout
    W_t = W.T.contiguous()
    ms = bench(lambda: torch._int_mm(x, W_t))
    M = M_actual   # restore for AI/mem calculation
    tflops = tflops_from_gemm(M, N, K, ms)
    # Memory: load x (INT8) + load W (INT8) + write out (INT32 = 4B)
    mem = mem_bytes_gemm(M, N, K, 1, 1, 4)
    ai = 2 * M * N * K / mem
    return ms, tflops, ai, mem

def run_int8_dequant(M, N, K):
    """INT8 weights → dequantize to BF16 → BF16 matmul.
    Simulates bitsandbytes LLM.int8() inference path.
    """
    W_fp = torch.randn(N, K, device=DEVICE)
    x    = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    W_i8 = (W_fp * 127 / W_fp.abs().max()).round().clamp(-128, 127).to(torch.int8)
    scale = W_fp.abs().max() / 127.0

    def _forward():
        # Step 1: dequantize INT8 → BF16  (the extra pass bitsandbytes adds)
        W_bf16 = W_i8.to(torch.bfloat16) * scale
        # Step 2: matmul in BF16
        return torch.nn.functional.linear(x, W_bf16)

    ms = bench(_forward)
    tflops = tflops_from_gemm(M, N, K, ms)
    # Memory: load W_i8 (INT8) + write W_bf16 (BF16) + read W_bf16 again (BF16) + load x (BF16) + write y
    # = K*N*1 + K*N*2 + K*N*2 + M*K*2 + M*N*2
    mem = K*N*1 + K*N*2 + K*N*2 + M*K*2 + M*N*2
    ai = 2 * M * N * K / mem
    return ms, tflops, ai, mem

def run_dequant_only(M, N, K):
    """Measure the dequantization kernel alone (INT8 → BF16 cast + scale).
    No matmul. Shows the pure overhead added by bitsandbytes-style path.
    """
    W_i8 = torch.randint(-128, 127, (N, K), device=DEVICE, dtype=torch.int8)
    scale = torch.tensor(0.01, device=DEVICE)

    def _dequant():
        return W_i8.to(torch.bfloat16) * scale

    ms = bench(_dequant)
    # Memory: read N*K INT8 bytes + write N*K BF16 bytes
    mem = N * K * 1 + N * K * 2
    flops = N * K   # one multiply per element
    ai = flops / mem
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops, ai, mem

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Quantization Memory Study — NVIDIA B200")
    print("  Isolating: bf16 / fp8_native / int8_native / int8_dequant / dequant_only")
    print("=" * 70)

    results = []
    paths = [
        ("bf16",         run_bf16,         DTYPE_PEAK["bf16"]),
        ("fp8_native",   run_fp8_native,    DTYPE_PEAK["fp8"]),
        ("int8_native",  run_int8_native,   DTYPE_PEAK["int8"]),
        ("int8_dequant", run_int8_dequant,  DTYPE_PEAK["bf16"]),  # effective ceil = bf16
    ]

    for M, N, K in SHAPES:
        print(f"\n── M={M:5d}  N={N}  K={K} ──")
        for label, fn, peak in paths:
            ms, tflops, ai, mem = fn(M, N, K)
            if ms is None:
                continue
            mem_bw_util = (mem / (ms * 1e-3)) / (MEM_BW_TBps * 1e12) * 100
            mfu = tflops / peak * 100
            print(f"  {label:16s}  {ms:7.3f} ms  {tflops:7.1f} TFLOPS  "
                  f"AI={ai:7.1f}  BW_util={mem_bw_util:5.1f}%  MFU={mfu:5.1f}%")
            results.append(dict(M=M, N=N, K=K, path=label,
                                ms=ms, tflops=tflops, ai=ai,
                                mem_bytes=mem, mem_bw_util=mem_bw_util, mfu=mfu))

    # Dequant-only: sweep weight matrix sizes (M is irrelevant — dequant only touches N×K weights)
    # Use sizes from ~10 MB up to ~4 GB to show how BW utilization rises with tensor size
    print(f"\n── Dequant-only kernel (INT8→BF16, no matmul) — sweep weight sizes ──")
    DQ_SHAPES = [
        (256,   512),    #   ~0.1 MB — kernel-overhead dominated
        (1024,  1024),   #   ~1 MB
        (2048,  4096),   #   ~8 MB
        (4096,  14336),  #  ~58 MB  — actual LLM FFN weight
        (8192,  14336),  # ~117 MB
        (8192,  28672),  # ~235 MB  — 70B LLM FFN
        (16384, 28672),  # ~470 MB
    ]
    dq_results = []
    for dq_N, dq_K in DQ_SHAPES:
        ms, tflops, ai, mem = run_dequant_only(1, dq_N, dq_K)
        bw_util = (mem / (ms * 1e-3)) / (MEM_BW_TBps * 1e12) * 100
        size_MB = dq_N * dq_K / 1e6
        print(f"  {dq_N}×{dq_K} ({size_MB:.0f} MB)  {ms:.3f} ms  "
              f"AI={ai:.3f}  BW_util={bw_util:.1f}%  "
              f"{'memory-bound' if ai < 281 else 'compute-bound'}")
        dq_results.append(dict(weight_MB=size_MB, N=dq_N, K=dq_K,
                               ms=ms, tflops=tflops, ai=ai, mem_bw_util=bw_util))

    df = pd.DataFrame(results)
    df_dq = pd.DataFrame(dq_results)
    df.to_csv(f"{OUT}/quant_paths_comparison.csv", index=False)
    df_dq.to_csv(f"{OUT}/dequant_only.csv", index=False)
    print(f"\n  Saved CSVs → {OUT}/")

    # ── Plot 1: Latency vs Batch Size (N=4096, K=14336) ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Linear Layer: BF16 vs FP8 Native vs INT8 Native vs INT8+Dequant\n"
                 f"N={N}, K={K} (LLM FFN projection)", fontsize=11)

    colors = {"bf16": "#2196F3", "fp8_native": "#4CAF50",
              "int8_native": "#FF9800", "int8_dequant": "#F44336"}

    for path, grp in df.groupby("path"):
        grp = grp.sort_values("M")
        c = colors.get(path, "gray")
        axes[0].plot(grp["M"], grp["ms"],       "o-", lw=2, label=path, color=c)
        axes[1].plot(grp["M"], grp["tflops"],   "o-", lw=2, label=path, color=c)
        axes[2].plot(grp["M"], grp["mem_bw_util"], "o-", lw=2, label=path, color=c)

    axes[0].set_xlabel("Batch Tokens (M)"); axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency"); axes[0].set_xscale("log"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Batch Tokens (M)"); axes[1].set_ylabel("TFLOPS")
    axes[1].set_title("Achieved TFLOPS"); axes[1].set_xscale("log"); axes[1].legend(fontsize=8)
    axes[2].set_xlabel("Batch Tokens (M)"); axes[2].set_ylabel("HBM BW Util (%)")
    axes[2].set_title("Memory BW Utilization"); axes[2].set_xscale("log"); axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f"{GOUT}/latency_vs_batch.png")

    # ── Plot 2: Roofline — multi-dtype ceilings, all-M trajectories ──────────
    # Each path has its own compute ceiling; show all M values as trajectories
    # to visualise the memory-bound → compute-bound transition per path.
    PATH_PEAK = {           # correct ceiling per execution path
        "bf16":         DTYPE_PEAK["bf16"],   # 2250 TFLOPS
        "fp8_native":   DTYPE_PEAK["fp8"],    # 4500 TFLOPS
        "int8_native":  DTYPE_PEAK["int8"],   # 4500 TOPS (same as fp8 numerically)
        "int8_dequant": DTYPE_PEAK["bf16"],   # dequant → bf16 matmul, capped at BF16
    }
    COLORS = {"bf16": "#2196F3", "fp8_native": "#4CAF50",
              "int8_native": "#FF9800", "int8_dequant": "#F44336",
              "dequant_only": "#9C27B0"}

    ai_range = np.logspace(-2, 4, 600)
    bw = MEM_BW_TBps

    fig, ax = plt.subplots(figsize=(11, 7))

    # Draw one roofline per distinct compute ceiling (avoid duplicate lines)
    drawn_peaks = set()
    for path, peak in PATH_PEAK.items():
        if peak in drawn_peaks: continue
        drawn_peaks.add(peak)
        roof = np.minimum(ai_range * bw, peak)
        label = f"{'BF16' if peak == DTYPE_PEAK['bf16'] else 'FP8/INT8'} roof ({peak:.0f} TFLOPS)"
        ax.loglog(ai_range, roof, lw=2, ls="-",
                  color="#2196F3" if peak == DTYPE_PEAK["bf16"] else "#4CAF50",
                  label=label)
        ridge = peak / bw
        ax.axvline(ridge, color="#2196F3" if peak == DTYPE_PEAK["bf16"] else "#4CAF50",
                   ls="--", lw=1, alpha=0.5)
        ax.text(ridge * 1.05, peak * 0.55,
                f"ridge={ridge:.0f}", fontsize=8,
                color="#2196F3" if peak == DTYPE_PEAK["bf16"] else "#4CAF50")

    # Memory slope label
    ax.text(0.05, bw * 0.05 * 1.4, f"Mem BW slope\n({bw:.0f} TB/s)",
            fontsize=8, color="gray", rotation=42)

    # Plot each matmul path as a trajectory across all M values
    for path, grp in df.groupby("path"):
        grp = grp.sort_values("M")
        c = COLORS.get(path, "gray")
        ax.loglog(grp["ai"], grp["tflops"], "o-", lw=1.8, ms=6,
                  color=c, label=path)
        # Annotate last point (highest M)
        last = grp.iloc[-1]
        ax.annotate(f"M={int(last['M'])}", (last["ai"], last["tflops"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7, color=c)

    # Dequant-only: single point (doesn't depend on M)
    dq_best = df_dq.loc[df_dq["weight_MB"].idxmax()]
    ax.scatter([dq_best["ai"]], [dq_best["tflops"]], marker="*", s=150,
               color=COLORS["dequant_only"], zorder=6, label="dequant_only (470 MB weight)")
    ax.annotate("dequant\nonly", (dq_best["ai"], dq_best["tflops"]),
                textcoords="offset points", xytext=(5, -12), fontsize=7,
                color=COLORS["dequant_only"])

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title("Roofline — Quantization Paths on NVIDIA B200\n"
                 "Trajectories show M=1→2048 batch tokens; each path has its own compute ceiling",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f"{GOUT}/roofline_quant_paths.png")

    # ── Plot 3: Dequant overhead — latency breakdown ──────────────────────────
    # For each M: bar showing int8_native vs int8_dequant, and the difference = dequant cost
    m_vals = sorted(df["M"].unique())
    x = np.arange(len(m_vals))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, path in enumerate(["bf16", "int8_native", "int8_dequant"]):
        sub = df[df["path"] == path].sort_values("M")
        ax.bar(x + i*w, sub["ms"].values, w, label=path, color=list(colors.values())[i], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"M={m}" for m in m_vals], rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Breakdown: where does INT8+dequant lose time?")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, f"{GOUT}/dequant_overhead_bar.png")

    # ── Plot 4: Dequant-only BW utilization vs weight matrix size ───────────────
    # Shows: small tensors are overhead-dominated; large tensors approach HBM peak
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    labels = [f"{r['N']}×{r['K']}\n({r['weight_MB']:.0f} MB)" for _, r in df_dq.iterrows()]

    axes[0].bar(range(len(df_dq)), df_dq["mem_bw_util"], color="#9C27B0", alpha=0.8)
    axes[0].axhline(100, color="red", lw=1.5, ls="--", label="100% HBM BW ceiling")
    axes[0].set_xticks(range(len(df_dq))); axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("HBM BW Utilization (%)")
    axes[0].set_title(f"Dequant Kernel (INT8→BF16): AI = {df_dq['ai'].iloc[0]:.2f} FLOP/byte\n"
                      "BW utilization rises with tensor size (overhead amortised)")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 110)

    # Latency breakdown: actual time vs ideal time at peak BW
    ideal_ms = df_dq["weight_MB"] * 3 / 1e3 / MEM_BW_TBps * 1e3   # (N*K*3 bytes) / BW
    axes[1].plot(df_dq["weight_MB"], df_dq["ms"], "o-", lw=2, color="#9C27B0", label="Actual latency")
    axes[1].plot(df_dq["weight_MB"], ideal_ms,    "s--", lw=2, color="green",   label="Ideal (peak HBM BW)")
    axes[1].set_xlabel("Weight Matrix Size (MB)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Actual vs Ideal Dequant Latency\n"
                      "Gap = kernel launch + memory transaction overhead")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, f"{GOUT}/dequant_bw_util.png")

    print(f"\n  Graphs → {GOUT}/")
    print("=" * 70)
    print("  Quantization Memory Study Complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
