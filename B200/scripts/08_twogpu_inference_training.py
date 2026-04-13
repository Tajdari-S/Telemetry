#!/usr/bin/env python3
"""
2-GPU vs 1-GPU Inference and Training Comparison
Shows how performance bottlenecks shift: memory-bound → NVLink-bound → compute-bound

Inference configs:
  1x GPU  — single B200
  2x GPU TP=2 — tensor-parallel via NVLink

Training configs:
  1x GPU — standard
  2x GPU — DataParallel (implicit NVLink for gradient sync)

Sweeps: batch_size, input_len, output_len, sequence_len
Tracks: throughput, TPOT, TTL, MFU, power, mem/GPU util, NVLink BW utilization
"""
import os, sys, time, json, gc, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.telemetry import TelemetryRecorder
from utils.roofline import plot_roofline, MEM_BW_TBps, B200_SPECS, DTYPE_PEAK
from utils.plotting import save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "twogpu")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "twogpu")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

N_GPUS = torch.cuda.device_count()

BATCH_SIZES = [1, 4, 8, 16, 32]
INPUT_LENS  = [128, 512, 1024, 2048]
OUTPUT_LEN  = 256
MODEL_PARAMS_B = 7.0  # Qwen 7B

# ── Inline LLaMA-style training model ────────────────────────────────────────
class LlamaFFN(nn.Module):
    def __init__(self, d, ffn_dim):
        super().__init__()
        self.gate = nn.Linear(d, ffn_dim, bias=False)
        self.up   = nn.Linear(d, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

class LlamaLayer(nn.Module):
    def __init__(self, d=4096, n_heads=32, n_kv=8, ffn_dim=14336):
        super().__init__()
        self.n_heads = n_heads; self.n_kv = n_kv; self.hd = d // n_heads
        self.attn_q = nn.Linear(d, d, bias=False)
        self.attn_k = nn.Linear(d, n_kv * self.hd, bias=False)
        self.attn_v = nn.Linear(d, n_kv * self.hd, bias=False)
        self.attn_o = nn.Linear(d, d, bias=False)
        self.ffn = LlamaFFN(d, ffn_dim)
        self.ln1 = nn.RMSNorm(d); self.ln2 = nn.RMSNorm(d)

    def forward(self, x):
        B, S, D = x.shape
        h, kv, hd = self.n_heads, self.n_kv, self.hd
        groups = h // kv
        xn = self.ln1(x)
        q = self.attn_q(xn).view(B, S, h, hd).transpose(1, 2)
        k = self.attn_k(xn).view(B, S, kv, hd).transpose(1, 2)
        v = self.attn_v(xn).view(B, S, kv, hd).transpose(1, 2)
        k = k.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(B, h, S, hd)
        v = v.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(B, h, S, hd)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.attn_o(a)
        return x + self.ffn(self.ln2(x))

class MiniLlama(nn.Module):
    def __init__(self, n_layers=8):
        super().__init__()
        self.embed  = nn.Embedding(32000, 4096)
        self.layers = nn.ModuleList([LlamaLayer() for _ in range(n_layers)])
        self.ln_f   = nn.RMSNorm(4096)
        self.head   = nn.Linear(4096, 32000, bias=False)
    def forward(self, ids):
        x = self.embed(ids)
        for l in self.layers: x = l(x)
        return self.head(self.ln_f(x))

def count_params(m): return sum(p.numel() for p in m.parameters()) / 1e9
def compute_mfu(tps, params_B, peak_TFLOPS):
    return 100 * tps * 6 * params_B * 1e9 / peak_TFLOPS / 1e12

def plot_comparison(df, x_col, y_col, group_col, title, ylabel, out_path):
    if df.empty or y_col not in df.columns: return
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp_val, grp in df.groupby(group_col):
        pivot = grp.groupby(x_col)[y_col].mean()
        ax.plot(pivot.index, pivot.values, "o-", lw=2, label=str(grp_val))
    ax.set_xlabel(x_col); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    save_fig(fig, out_path)

# ── Inference sweep ───────────────────────────────────────────────────────────
def run_inference_sweep(model_name, dtypes=("bf16", "fp8")):
    from vllm import LLM, SamplingParams
    results = []
    words = ["the", "quick", "brown", "fox", "over", "lazy", "dog",
             "is", "a", "an", "to", "of", "and", "with", "for"]

    configs = [(1, "1xGPU")] + ([(2, "2xGPU_TP")] if N_GPUS >= 2 else [])

    for num_gpus, config_label in configs:
        for dtype_label in dtypes:
            quant = "fp8" if dtype_label == "fp8" else None
            print(f"\n  [{config_label} | {dtype_label}] Loading engine...")
            try:
                engine = LLM(
                    model=model_name,
                    dtype="bfloat16",
                    quantization=quant,
                    tensor_parallel_size=num_gpus,
                    max_model_len=4096,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.88,
                )
            except Exception as e:
                print(f"    FAIL: {e}")
                continue

            gpu_ids = list(range(num_gpus))
            for in_len in INPUT_LENS:
                for bs in BATCH_SIZES:
                    prompt  = " ".join(np.random.choice(words, in_len))
                    prompts = [prompt] * bs
                    sp = SamplingParams(max_tokens=OUTPUT_LEN, temperature=0.0)
                    rec = TelemetryRecorder(gpu_indices=gpu_ids, hz=20)
                    rec.start()
                    t0  = time.perf_counter()
                    outs = engine.generate(prompts, sp)
                    t1  = time.perf_counter()
                    df_t = rec.stop()
                    elapsed = t1 - t0
                    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
                    tps     = out_tok / elapsed
                    tpot    = elapsed / max(out_tok, 1) * 1000
                    rows    = df_t[df_t["gpu_idx"].isin(gpu_ids)]
                    results.append({
                        "config": config_label, "dtype": dtype_label,
                        "num_gpus": num_gpus, "batch_size": bs,
                        "input_len": in_len, "output_len": OUTPUT_LEN,
                        "throughput_tps": tps, "tpot_ms": tpot,
                        "ttl_s": elapsed,
                        "mean_power_W":    float(rows["power_w"].mean()),
                        "mem_bw_util_pct": float(rows["mem_util"].mean()),
                        "gpu_util_pct":    float(rows["gpu_util"].mean()),
                        "mean_sm_clock":   float(rows["sm_clock_mhz"].mean()),
                    })
                    print(f"    bs={bs:3d} in={in_len:5d}: {tps:7.0f} tok/s  "
                          f"TPOT={tpot:.1f}ms  pwr={results[-1]['mean_power_W']:.0f}W")
            del engine; gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{OUT}/inference_1x_vs_2x.csv", index=False)
    return df

# ── Training sweep ────────────────────────────────────────────────────────────
def run_training_sweep():
    results = []
    configs = [(1, "1xGPU")] + ([(2, "2xGPU_DP")] if N_GPUS >= 2 else [])
    peak_BF16 = DTYPE_PEAK["bf16"]

    for num_gpus, config_label in configs:
        device  = torch.device("cuda:0")
        gpu_ids = list(range(num_gpus))
        print(f"\n  [{config_label}] Building model...")
        model = MiniLlama(n_layers=8).to(device).to(torch.bfloat16)
        if num_gpus > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
        actual_B = count_params(model)
        opt = optim.AdamW(model.parameters(), lr=1e-4)
        ce  = nn.CrossEntropyLoss()

        for seq in [128, 512, 1024, 2048]:
            for bs in [1, 2, 4, 8]:
                try:
                    ids    = torch.randint(0, 32000, (bs, seq), device=device)
                    target = torch.randint(0, 32000, (bs, seq), device=device)
                    # Warmup
                    for _ in range(2):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = ce(model(ids).view(-1, 32000), target.view(-1))
                        loss.backward(); opt.step(); opt.zero_grad()
                    torch.cuda.synchronize()
                    rec = TelemetryRecorder(gpu_indices=gpu_ids, hz=20)
                    rec.start()
                    t0 = time.perf_counter()
                    N_STEPS = 5
                    for _ in range(N_STEPS):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = ce(model(ids).view(-1, 32000), target.view(-1))
                        loss.backward(); opt.step(); opt.zero_grad()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    df_t = rec.stop()
                    tps   = bs * seq * N_STEPS / elapsed
                    scaled= tps * (MODEL_PARAMS_B / actual_B)
                    mfu   = compute_mfu(scaled, MODEL_PARAMS_B, peak_BF16)
                    rows  = df_t[df_t["gpu_idx"].isin(gpu_ids)]
                    results.append({
                        "config": config_label, "num_gpus": num_gpus,
                        "batch_size": bs, "seq_len": seq,
                        "tokens_per_sec": tps, "scaled_tps": scaled,
                        "mfu_pct": mfu, "loss": float(loss.item()),
                        "mean_power_W":    float(rows["power_w"].mean()),
                        "mem_bw_util_pct": float(rows["mem_util"].mean()),
                        "gpu_util_pct":    float(rows["gpu_util"].mean()),
                    })
                    print(f"    bs={bs:3d} seq={seq:5d}: {tps:7.0f} tok/s  "
                          f"MFU={mfu:.1f}%  pwr={results[-1]['mean_power_W']:.0f}W")
                except torch.cuda.OutOfMemoryError:
                    print(f"    bs={bs:3d} seq={seq:5d}: OOM"); torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    bs={bs:3d} seq={seq:5d}: {e}")
        del model, opt; gc.collect(); torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(f"{OUT}/training_1x_vs_2x.csv", index=False)
    return df

# ── Bottleneck analysis plot ──────────────────────────────────────────────────
def plot_bottleneck_analysis(df, task_label, out_path):
    """Show how GPU util, mem BW util, and power shift with config and batch."""
    if df.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Bottleneck Analysis — {task_label}: 1x vs 2x GPU", fontsize=12)
    metrics = [("gpu_util_pct","GPU Util (%)"), ("mem_bw_util_pct","Mem BW Util (%)"),
               ("mean_power_W","Power (W)")]
    for ax, (met, lbl) in zip(axes, metrics):
        for cfg, grp in df.groupby("config"):
            x_col = "batch_size" if "batch_size" in df.columns else "seq_len"
            pivot = grp.groupby(x_col)[met].mean()
            ax.plot(pivot.index, pivot.values, "o-", lw=2, label=cfg)
        ax.set_xlabel("Batch Size"); ax.set_ylabel(lbl)
        ax.set_title(lbl); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtypes", default="bf16,fp8")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-training",  action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  2-GPU vs 1-GPU Comparison — NVIDIA B200")
    print(f"  GPUs detected: {N_GPUS}")
    print("=" * 70)

    if not args.skip_inference:
        print("\n━━ INFERENCE SWEEP ━━")
        df_inf = run_inference_sweep(args.model, args.dtypes.split(","))
        if not df_inf.empty:
            for dtype_label in df_inf["dtype"].unique():
                d = df_inf[df_inf["dtype"] == dtype_label]
                plot_comparison(
                    d[d["input_len"] == 128], "batch_size", "throughput_tps", "config",
                    f"Throughput vs Batch Size [{dtype_label}]",
                    "tok/s", f"{GOUT}/inf_{dtype_label}_tps_vs_bs.png")
                plot_comparison(
                    d[d["batch_size"] == 8], "input_len", "throughput_tps", "config",
                    f"Throughput vs Input Len [{dtype_label}]",
                    "tok/s", f"{GOUT}/inf_{dtype_label}_tps_vs_inlen.png")
                plot_bottleneck_analysis(d, f"Inference {dtype_label}",
                    f"{GOUT}/inf_{dtype_label}_bottleneck.png")
                # TPOT vs batch
                plot_comparison(
                    d[d["input_len"] == 128], "batch_size", "tpot_ms", "config",
                    f"TPOT vs Batch Size [{dtype_label}]",
                    "TPOT (ms)", f"{GOUT}/inf_{dtype_label}_tpot_vs_bs.png")

    if not args.skip_training:
        print("\n━━ TRAINING SWEEP ━━")
        df_train = run_training_sweep()
        if not df_train.empty:
            plot_comparison(
                df_train[df_train["seq_len"] == 512], "batch_size", "tokens_per_sec", "config",
                "Training Throughput vs Batch Size (BF16)",
                "tok/s", f"{GOUT}/train_bf16_tps_vs_bs.png")
            plot_comparison(
                df_train[df_train["batch_size"] == 4], "seq_len", "mfu_pct", "config",
                "MFU vs Seq Length — 1x vs 2x GPU",
                "MFU (%)", f"{GOUT}/train_bf16_mfu_vs_seqlen.png")
            plot_comparison(
                df_train[df_train["batch_size"] == 4], "seq_len", "mean_power_W", "config",
                "Power vs Seq Length — 1x vs 2x GPU",
                "Power (W)", f"{GOUT}/train_bf16_power_vs_seqlen.png")
            plot_bottleneck_analysis(
                df_train.rename(columns={"seq_len": "batch_size"}),
                "Training BF16",
                f"{GOUT}/train_bf16_bottleneck.png")

    print("\n" + "=" * 70)
    print("  2-GPU Comparison Complete.")
    print(f"  Results → {OUT}/  |  Graphs → {GOUT}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
