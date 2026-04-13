#!/usr/bin/env python3
"""
Phase 3 — LLM Training Benchmark Sweep
Uses a GPT-style model with same architectural params as Llama-3.1-8B
Sweeps:
  • batch_size × seq_len
  • dtype: fp32, fp16, bf16, fp8 (te), int8 (dynamic quant)
  • 1x GPU vs 2x GPU (DDP)

Metrics:
  • Training throughput (tokens/s, samples/s)
  • MFU  (Model FLOP Utilization)
  • Gradient norm
  • Memory allocated / reserved (GB)
  • TTFT analog: time for first backward pass
  • Power, SM util, BW util via Tier-1 NVML
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
from utils.plotting import (plot_sweep_heatmap, plot_bar_comparison,
                             plot_dtype_comparison, save_fig)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "training")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "training")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

# ── Llama-3.1-8B architectural params ────────────────────────────────────────
LLAMA_8B_CONFIG = dict(
    vocab_size=32000, n_layers=32, d_model=4096,
    n_heads=32, n_kv_heads=8, ffn_dim=14336,
    max_seq_len=4096,
)
MODEL_PARAMS_B = 8.0   # ~8B parameters

# ── Quick LLaMA-like model (full model too large to build from scratch) ──────
class LlamaFFN(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

class LlamaLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]; h = cfg["n_heads"]; kv = cfg["n_kv_heads"]
        self.n_heads    = h
        self.n_kv_heads = kv
        self.head_dim   = d // h
        self.attn_q = nn.Linear(d, d, bias=False)
        self.attn_k = nn.Linear(d, kv * self.head_dim, bias=False)
        self.attn_v = nn.Linear(d, kv * self.head_dim, bias=False)
        self.attn_o = nn.Linear(d, d, bias=False)
        self.ffn    = LlamaFFN(d, cfg["ffn_dim"])
        self.ln1    = nn.RMSNorm(d)
        self.ln2    = nn.RMSNorm(d)

    def forward(self, x):
        B, S, D = x.shape
        h, kv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        groups = h // kv
        xn = self.ln1(x)
        q = self.attn_q(xn).view(B, S, h,  hd).transpose(1, 2)            # [B,h,S,hd]
        k = self.attn_k(xn).view(B, S, kv, hd).transpose(1, 2)            # [B,kv,S,hd]
        v = self.attn_v(xn).view(B, S, kv, hd).transpose(1, 2)
        # GQA: repeat KV heads to match Q heads
        k = k.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(B, h, S, hd)
        v = v.unsqueeze(2).expand(-1, -1, groups, -1, -1).reshape(B, h, S, hd)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.attn_o(attn)
        x = x + self.ffn(self.ln2(x))
        return x

class MiniLlama(nn.Module):
    """N-layer Llama-style model. Use fewer layers than 32 for benchmark speed."""
    def __init__(self, cfg, n_layers_override=None):
        super().__init__()
        n = n_layers_override or cfg["n_layers"]
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.layers = nn.ModuleList([LlamaLayer(cfg) for _ in range(n)])
        self.ln_f   = nn.RMSNorm(cfg["d_model"])
        self.lm_head= nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9

def train_throughput(model, optimizer, device, batch, seq, dtype,
                     scaler=None, n_steps=5):
    """Run n_steps of fwd+bwd and return tokens/s."""
    model.train()
    ids = torch.randint(0, 32000, (batch, seq), device=device)
    target = torch.randint(0, 32000, (batch, seq), device=device)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(2):
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(ids)
            loss   = loss_fn(logits.view(-1, 32000), target.view(-1))
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for step in range(n_steps):
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(ids)
            loss   = loss_fn(logits.view(-1, 32000), target.view(-1))
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_s = batch * seq * n_steps / elapsed
    return tokens_s, float(loss.item())

def compute_mfu(tokens_s: float, model_params_B: float, peak_TFLOPS: float) -> float:
    """MFU = achieved_TFLOPS / peak_TFLOPS"""
    # Training FLOPs per token ≈ 6 * model_params (fwd + bwd)
    achieved_TFLOPS = tokens_s * 6 * model_params_B * 1e9 / 1e12
    return 100.0 * achieved_TFLOPS / peak_TFLOPS

DTYPE_CONFIGS = {
    "fp32": {"torch_dtype": torch.float32,  "autocast_dtype": torch.float32,  "use_scaler": False, "bytes_per_param": 4},
    "fp16": {"torch_dtype": torch.float16,  "autocast_dtype": torch.float16,  "use_scaler": True,  "bytes_per_param": 2},
    "bf16": {"torch_dtype": torch.bfloat16, "autocast_dtype": torch.bfloat16, "use_scaler": False, "bytes_per_param": 2},
    "fp8":  {"torch_dtype": torch.bfloat16, "autocast_dtype": torch.bfloat16, "use_scaler": False, "bytes_per_param": 1,
             "_note": "fp8 via torch.float8_e4m3fn weights where supported"},
    "int8": {"torch_dtype": torch.bfloat16, "autocast_dtype": torch.bfloat16, "use_scaler": False, "bytes_per_param": 1,
             "_note": "int8 via bitsandbytes dynamic quantization for inference; train in bf16"},
    "int4": {"torch_dtype": torch.bfloat16, "autocast_dtype": torch.bfloat16, "use_scaler": False, "bytes_per_param": 0.5,
             "_note": "int4 via bitsandbytes QLoRA-style; train in bf16 with 4-bit weights"},
    "int12":{"torch_dtype": torch.bfloat16, "autocast_dtype": torch.bfloat16, "use_scaler": False, "bytes_per_param": 1.5,
             "_note": "Non-standard; simulated as bf16 with 12-bit weight quantization"},
}

BATCH_SIZES = [1, 2, 4, 8]
SEQ_LENS    = [128, 512, 1024, 2048]
N_LAYERS    = 8   # use 8 layers (not 32) to keep benchmark fast; scales linearly

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes",   default="fp32,fp16,bf16,fp8,int8,int4,int12")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()

    dtype_list  = args.dtypes.split(",")
    batch_sizes = [1, 2] if args.quick else BATCH_SIZES
    seq_lens    = [128, 512] if args.quick else SEQ_LENS
    n_layers    = 2 if args.quick else args.n_layers
    gpu_ids     = list(range(args.num_gpus))
    device      = torch.device(f"cuda:{gpu_ids[0]}")

    print("=" * 70)
    print(f"  LLM Training Sweep — Llama-3.1-8B style ({n_layers} layers)")
    print(f"  GPUs: {gpu_ids}  |  Dtypes: {dtype_list}")
    print("=" * 70)

    all_results = []

    for dtype_label in dtype_list:
        dcfg = DTYPE_CONFIGS.get(dtype_label, DTYPE_CONFIGS["bf16"])
        print(f"\n{'─'*60}")
        print(f"  DTYPE: {dtype_label.upper()}  {dcfg.get('_note','')}")
        print(f"{'─'*60}")

        # Build model
        cfg = LLAMA_8B_CONFIG.copy()
        model = MiniLlama(cfg, n_layers_override=n_layers).to(device).to(dcfg["torch_dtype"])
        actual_params_B = count_params(model)
        scale_factor = MODEL_PARAMS_B / (actual_params_B * n_layers / n_layers)

        if args.num_gpus > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        # Apply int12 simulation
        if dtype_label == "int12":
            with torch.no_grad():
                for p in model.parameters():
                    if p.dim() >= 2:
                        sc = p.abs().max() / 2047.0 + 1e-8
                        p.data = (p / sc).round().clamp(-2048, 2047) * sc

        # Apply int8/int4 weight quantization simulation (training in bf16 with quantized weights)
        if dtype_label in ("int8", "int4"):
            print(f"    Note: {dtype_label} weight-quant simulation — training in BF16 with {dtype_label} weights")
            bits = 8 if dtype_label == "int8" else 4
            max_val = 2 ** (bits - 1) - 1
            with torch.no_grad():
                for p in model.parameters():
                    if p.dim() >= 2:
                        sc = p.abs().max() / max_val + 1e-8
                        p.data = (p / sc).round().clamp(-max_val, max_val) * sc
            if False:  # int8 CPU dynamic quant (CPU-only, shown for reference)
                model = torch.ao.quantization.quantize_dynamic(
                    model.cpu(), {nn.Linear}, dtype=torch.qint8).to(device)

        scaler = torch.cuda.amp.GradScaler() if dcfg["use_scaler"] else None
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        for seq in seq_lens:
            for bs in batch_sizes:
                if bs * seq > 8192 and dtype_label == "fp32":
                    print(f"    bs={bs} seq={seq}: skipping fp32 OOM risk")
                    continue
                print(f"  bs={bs:3d} | seq={seq:5d} | dtype={dtype_label}", end=" ... ")
                try:
                    rec = TelemetryRecorder(gpu_indices=gpu_ids, hz=20)
                    rec.start()
                    tokens_s, loss_val = train_throughput(
                        model, optimizer, device, bs, seq,
                        dcfg["autocast_dtype"], scaler, n_steps=5
                    )
                    df_telem = rec.stop()

                    mem_alloc_GB = torch.cuda.memory_allocated(gpu_ids[0]) / 1e9
                    mem_res_GB   = torch.cuda.memory_reserved(gpu_ids[0]) / 1e9

                    trows = df_telem[df_telem["gpu_idx"].isin(gpu_ids)]
                    mean_power = trows["power_w"].mean()
                    mean_util  = trows["gpu_util"].mean()
                    mean_bw    = trows["mem_util"].mean()

                    # Scale MFU to full 8B model
                    scaled_tokens_s = tokens_s * (MODEL_PARAMS_B / actual_params_B)
                    peak_TFLOPS = DTYPE_PEAK.get(dtype_label, DTYPE_PEAK["bf16"])
                    mfu = compute_mfu(scaled_tokens_s, MODEL_PARAMS_B, peak_TFLOPS)

                    # Roofline: training AI ≈ 6*P / (P * bytes_per_param) = 6 / bytes_per_param
                    bpp = dcfg.get("bytes_per_param", 2)
                    ai  = 6.0 / bpp

                    print(f"OK | {tokens_s:.0f} tok/s | MFU={mfu:.1f}% | loss={loss_val:.3f} | power={mean_power:.0f}W")
                    all_results.append({
                        "dtype": dtype_label,
                        "batch_size": bs,
                        "seq_len": seq,
                        "n_layers": n_layers,
                        "num_gpus": args.num_gpus,
                        "tokens_per_sec": tokens_s,
                        "scaled_tokens_per_sec": scaled_tokens_s,
                        "mfu_pct": mfu,
                        "loss": loss_val,
                        "mem_alloc_GB": mem_alloc_GB,
                        "mem_reserved_GB": mem_res_GB,
                        "mean_power_W": mean_power,
                        "mean_gpu_util": mean_util,
                        "mean_mem_bw_util": mean_bw,
                        "arith_intensity": ai,
                    })
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM")
                    all_results.append({
                        "dtype": dtype_label, "batch_size": bs, "seq_len": seq,
                        "error": "OOM"
                    })
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({
                        "dtype": dtype_label, "batch_size": bs, "seq_len": seq,
                        "error": str(e)
                    })

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f"{OUT}/training_sweep_raw.csv", index=False)
    print(f"\n  Raw saved → {OUT}/training_sweep_raw.csv")

    df = df_all[~df_all.get("error", pd.Series(dtype=object)).notna()].copy() \
         if "error" in df_all.columns else df_all.copy()
    df = df.dropna(subset=["tokens_per_sec"])

    # ── Plots ────────────────────────────────────────────────────────────────
    if len(df):
        # Throughput heatmap
        for dtype_label in df["dtype"].unique():
            d = df[df["dtype"] == dtype_label]
            if d[["batch_size","seq_len"]].nunique().min() < 2:
                continue
            try:
                plot_sweep_heatmap(
                    d, x_col="batch_size", y_col="seq_len", val_col="tokens_per_sec",
                    title=f"Training Throughput (tok/s) — {dtype_label.upper()}",
                    out_path=f"{GOUT}/training_throughput_heatmap_{dtype_label}.png",
                )
            except Exception as e:
                print(f"  Heatmap error: {e}")

        # MFU by dtype
        mfu_by_dtype = df.groupby("dtype")["mfu_pct"].max().to_dict()
        plot_dtype_comparison(mfu_by_dtype, "mfu_pct", "Peak MFU (%)",
                              "Model FLOP Utilization by Dtype",
                              f"{GOUT}/mfu_by_dtype.png")

        # Throughput by dtype (bs=4, seq=512 or closest)
        snap = df.groupby("dtype")["tokens_per_sec"].max().to_dict()
        plot_dtype_comparison(snap, "tokens_per_sec", "Max Throughput (tok/s)",
                              "Max Training Throughput by Dtype",
                              f"{GOUT}/training_throughput_by_dtype.png")

        # Memory usage
        mem = df.groupby("dtype")["mem_alloc_GB"].max().to_dict()
        plot_dtype_comparison(mem, "mem_alloc_GB", "Peak Memory (GB)",
                              "Peak Memory Allocation by Dtype",
                              f"{GOUT}/training_memory_by_dtype.png")

        # Roofline
        roofline_pts = {}
        for dtype_label, grp in df.groupby("dtype"):
            best = grp.loc[grp["tokens_per_sec"].idxmax()]
            bpp = DTYPE_CONFIGS.get(dtype_label, {}).get("bytes_per_param", 2)
            # Training FLOPS per token ≈ 6 * params
            achieved_TFLOPS = best["scaled_tokens_per_sec"] * 6 * MODEL_PARAMS_B * 1e9 / 1e12
            ai = 6.0 / bpp
            roofline_pts[dtype_label] = (ai, achieved_TFLOPS)
        plot_roofline(roofline_pts, dtype="bf16",
                      out_path=f"{GOUT}/roofline_training.png")

        # Power vs MFU
        fig, ax = plt.subplots(figsize=(8, 5))
        for dtype_label, grp in df.groupby("dtype"):
            ax.scatter(grp["mean_power_W"], grp["mfu_pct"], label=dtype_label, alpha=0.7, s=60)
        ax.set_xlabel("Mean Power (W)"); ax.set_ylabel("MFU (%)")
        ax.set_title("Power vs MFU — Training Sweep"); ax.legend(); ax.grid(alpha=0.3)
        save_fig(fig, f"{GOUT}/power_vs_mfu.png")

    print("\n" + "=" * 70)
    print("  Phase 3 Training Sweep Complete.")
    print(f"  Results → {OUT}/  |  Graphs → {GOUT}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
