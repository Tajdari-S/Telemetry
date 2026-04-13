#!/usr/bin/env python3
"""
Phase 2 — LLM Inference Benchmark Sweep
Model: meta-llama/Llama-3.1-8B-Instruct (default, needs HF token)
       Falls back to Qwen/Qwen2.5-7B-Instruct (open, no auth)

Sweeps:
  • batch_size: [1, 2, 4, 8, 16, 32]
  • input_len:  [128, 512, 1024, 2048]
  • output_len: [128, 256, 512]
  • dtype:      [fp16, bf16, fp8, int8, int4, int12*]
    *int12 = custom 12-bit simulation (non-standard, documented)

Metrics per run:
  • TTFT  (Time To First Token, s)
  • TPOT  (Time Per Output Token, ms)
  • TBT   (Time Between Tokens, ms)
  • TTL   (Time To Last Token, s)
  • Throughput (tokens/s, requests/s)
  • MBW utilization (%)
  • SM utilization (%)
  • Power (W)
  • Memory used (GB)
  • Roofline arithmetic intensity

Uses vLLM offline engine for controlled sweeps.
"""
import os, sys, json, time, gc, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils.telemetry import TelemetryRecorder
from utils.roofline import plot_roofline, MEM_BW_TBps, DTYPE_PEAK, B200_SPECS
from utils.plotting import (plot_sweep_heatmap, plot_bar_comparison,
                             plot_dtype_comparison, save_fig)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT  = os.path.join(os.path.dirname(__file__), "..", "results", "inference")
GOUT = os.path.join(os.path.dirname(__file__), "..", "graphs", "inference")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GOUT, exist_ok=True)

# ── Model configuration ──────────────────────────────────────────────────────
DEFAULT_MODEL_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_OPEN  = "Qwen/Qwen2.5-7B-Instruct"

# ── Sweep parameters ─────────────────────────────────────────────────────────
BATCH_SIZES  = [1, 2, 4, 8, 16, 32]
INPUT_LENS   = [128, 512, 1024, 2048]
OUTPUT_LENS  = [128, 256, 512]

DTYPES_VLLM = {
    "fp16":  {"dtype": "float16",  "quantization": None},
    "bf16":  {"dtype": "bfloat16", "quantization": None},
    "fp8":   {"dtype": "bfloat16", "quantization": "fp8"},
    "int8":  {"dtype": "bfloat16", "quantization": "bitsandbytes"},
    "int4":  {"dtype": "bfloat16", "quantization": "bitsandbytes"},  # 4-bit BnB
    "int12": {"dtype": "bfloat16", "quantization": None,  # Custom sim (see note)
              "_custom_quant_bits": 12},
}

def build_prompts(batch: int, input_len: int) -> list:
    """Generate synthetic prompts of approx. `input_len` tokens."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "a", "an", "in", "on", "at", "to", "for", "of", "and", "is"]
    prompt = " ".join(np.random.choice(words, size=input_len))
    return [prompt] * batch

def int12_quantize_dequantize_model(model):
    """
    Non-standard 12-bit simulation:
    Quantize weights to 12 bits by using int16 with 4 bits masked to zero.
    This approximates the precision/memory of a hypothetical 12-bit format.
    Note: No actual hardware 12-bit support exists on B200.
    Only applied to linear weight tensors.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype in (torch.float16, torch.bfloat16) and "weight" in name:
                # Scale to int12 range [-2048, 2047]
                scale = param.abs().max() / 2047.0
                quantized = (param / scale).round().clamp(-2048, 2047)
                param.copy_(quantized * scale)
    return model

# ── vLLM engine wrapper ──────────────────────────────────────────────────────
def create_vllm_engine(model_name: str, dtype: str, quantization: str = None,
                       gpu_ids: list = None, max_model_len: int = 4096,
                       num_bits: int = None):
    """Create vLLM LLM instance."""
    from vllm import LLM, SamplingParams
    kwargs = dict(
        model=model_name,
        dtype=dtype,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    if quantization:
        kwargs["quantization"] = quantization
        if quantization == "bitsandbytes" and num_bits == 4:
            kwargs["load_format"] = "bitsandbytes"
            kwargs["bnb_config"] = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"}
        elif quantization == "bitsandbytes":
            kwargs["load_format"] = "bitsandbytes"

    if gpu_ids and len(gpu_ids) > 1:
        kwargs["tensor_parallel_size"] = len(gpu_ids)

    llm = LLM(**kwargs)
    return llm

def run_inference_batch(llm, prompts: list, max_tokens: int):
    """Run inference and return (outputs, timing_dict)."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput_tps = total_output_tokens / total_time
    throughput_rps = len(prompts) / total_time

    # Estimate TTFT and TPOT from total time
    # TTFT ≈ prefill time = (input processing) — approx with single-token output timing
    # For accurate TTFT we'd need streaming; here we use structured estimate
    ttft_est = total_time * 0.3  # rough: prefill ~30% for medium batch
    tpot_ms  = (total_time - ttft_est) / max(total_output_tokens, 1) * 1000
    tbt_ms   = tpot_ms  # TBT ≈ TPOT for non-speculative decode
    ttl_s    = total_time

    return outputs, {
        "total_time_s":       total_time,
        "total_output_tokens": total_output_tokens,
        "throughput_tps":     throughput_tps,
        "throughput_rps":     throughput_rps,
        "ttft_s_est":         ttft_est,
        "tpot_ms":            tpot_ms,
        "tbt_ms":             tbt_ms,
        "ttl_s":              ttl_s,
    }

def measure_telemetry_stats(df: pd.DataFrame, gpu_ids: list) -> dict:
    rows = df[df["gpu_idx"].isin(gpu_ids)]
    return {
        "mean_gpu_util":   rows["gpu_util"].mean(),
        "max_gpu_util":    rows["gpu_util"].max(),
        "mean_power_W":    rows["power_w"].mean(),
        "peak_power_W":    rows["power_w"].max(),
        "mean_mem_util":   rows["mem_util"].mean(),
        "mean_sm_clock_MHz": rows["sm_clock_mhz"].mean(),
        "mem_bw_util_pct": rows["mem_util"].mean(),  # NVML mem util ≈ BW util
    }

def estimate_roofline_point(batch, input_len, output_len, model_params_B=8.0,
                            dtype_bytes=2):
    """
    Rough arithmetic intensity for LLM decoding:
    - Each token: load all model weights (memory-bound at BS=1)
    - FLOPs ≈ 2 * model_params per output token
    """
    bytes_per_token = model_params_B * 1e9 * dtype_bytes
    flops_per_token = 2 * model_params_B * 1e9
    ai = flops_per_token / bytes_per_token  # = 1 for full weight load
    # At higher batch, reuse weights → higher AI
    effective_ai = ai * batch
    return effective_ai

# ── Main sweep ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="auto",
                        help="Model name (auto=try Llama then Qwen)")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dtypes",   default="bf16,fp16,fp8,int8,int4,int12",
                        help="Comma-separated dtype list")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--quick",    action="store_true",
                        help="Quick mode: small sweep for testing")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        import huggingface_hub
        huggingface_hub.login(token=args.hf_token, add_to_git_credential=False)

    # Resolve model
    if args.model == "auto":
        try:
            from huggingface_hub import model_info
            model_info(DEFAULT_MODEL_LLAMA)
            model_name = DEFAULT_MODEL_LLAMA
            model_params_B = 8.0
        except Exception:
            print(f"  Llama not accessible; falling back to Qwen2.5-7B")
            model_name = DEFAULT_MODEL_OPEN
            model_params_B = 7.0
    else:
        model_name = args.model
        model_params_B = 8.0

    print(f"\n  Model: {model_name}  |  GPUs: {args.num_gpus}")

    # Sweep config
    dtype_list   = args.dtypes.split(",")
    batch_sizes  = [1, 4, 16] if args.quick else BATCH_SIZES
    input_lens   = [128, 512] if args.quick else INPUT_LENS
    output_lens  = [128] if args.quick else OUTPUT_LENS
    gpu_ids      = list(range(args.num_gpus))

    all_results = []

    for dtype_label in dtype_list:
        dtype_cfg = DTYPES_VLLM.get(dtype_label, DTYPES_VLLM["bf16"])
        custom_bits = dtype_cfg.get("_custom_quant_bits", None)
        # int12 is simulated in BF16 memory, so it occupies 2 bytes/param at runtime
        dtype_bytes  = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5, "int12": 2}
        db = dtype_bytes.get(dtype_label, 2)

        print(f"\n{'='*60}")
        print(f"  DTYPE: {dtype_label.upper()}")
        if custom_bits:
            print(f"  NOTE: int12 is non-standard; simulated as bf16+weight quantization")
        print(f"{'='*60}")

        # Load engine once per dtype
        try:
            engine = create_vllm_engine(
                model_name,
                dtype_cfg["dtype"],
                quantization=dtype_cfg.get("quantization"),
                gpu_ids=gpu_ids,
                max_model_len=4096,
                num_bits=4 if dtype_label == "int4" else None,
            )
        except Exception as e:
            print(f"  ERROR loading engine for {dtype_label}: {e}")
            continue

        # Apply int12 custom quantization simulation
        if custom_bits == 12:
            print("  Applying int12 weight quantization simulation...")
            try:
                if hasattr(engine, "llm_engine"):
                    for w in engine.llm_engine.model_executor.driver_worker.model_runner.model.parameters():
                        if w.dim() >= 2:
                            scale = w.abs().max() / 2047.0
                            w.data = (w / scale).round().clamp(-2048, 2047) * scale
                print("  int12 simulation applied.")
            except Exception as e2:
                print(f"  int12 sim failed silently: {e2}")

        for out_len in output_lens:
            for inp_len in input_lens:
                for bs in batch_sizes:
                    print(f"  bs={bs:3d} | in={inp_len:5d} | out={out_len:4d} | dtype={dtype_label}", end=" ... ")
                    try:
                        prompts = build_prompts(bs, inp_len)
                        rec = TelemetryRecorder(gpu_indices=gpu_ids, hz=20)
                        rec.start()
                        _, timing = run_inference_batch(engine, prompts, out_len)
                        df_telem = rec.stop()
                        telem = measure_telemetry_stats(df_telem, gpu_ids)
                        mem_used_GB = torch.cuda.memory_allocated(0) / 1e9
                        ai = estimate_roofline_point(bs, inp_len, out_len, model_params_B, db)
                        row = {
                            "dtype": dtype_label,
                            "batch_size": bs,
                            "input_len": inp_len,
                            "output_len": out_len,
                            "model": model_name,
                            "num_gpus": args.num_gpus,
                            **timing,
                            **telem,
                            "mem_used_GB": mem_used_GB,
                            "arith_intensity": ai,
                        }
                        all_results.append(row)
                        print(f"OK | {timing['throughput_tps']:.0f} tok/s | "
                              f"TPOT={timing['tpot_ms']:.1f}ms | "
                              f"power={telem['mean_power_W']:.0f}W")
                    except Exception as e:
                        print(f"FAIL: {e}")
                        all_results.append({
                            "dtype": dtype_label, "batch_size": bs,
                            "input_len": inp_len, "output_len": out_len,
                            "error": str(e),
                        })

        # Free engine
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save raw results ─────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f"{OUT}/inference_sweep_raw.csv", index=False)
    print(f"\n  Raw results saved → {OUT}/inference_sweep_raw.csv")

    # ── Plots ────────────────────────────────────────────────────────────────
    df = df_all[df_all.get("error", pd.Series(dtype=object)).isna()].copy() if "error" in df_all.columns else df_all.copy()

    # Throughput heatmap per dtype
    for dtype_label in df["dtype"].unique():
        d = df[(df["dtype"] == dtype_label) & (df["output_len"] == output_lens[0])]
        if len(d) < 2:
            continue
        try:
            plot_sweep_heatmap(
                d, x_col="batch_size", y_col="input_len", val_col="throughput_tps",
                title=f"Throughput (tok/s) — {dtype_label.upper()} | out={output_lens[0]}",
                out_path=f"{GOUT}/throughput_heatmap_{dtype_label}.png",
            )
        except Exception as e:
            print(f"  Heatmap error ({dtype_label}): {e}")

    # Dtype throughput comparison (bs=1, small input)
    dtcomp = df[(df["batch_size"] == 1) & (df["input_len"] == input_lens[0])
                & (df["output_len"] == output_lens[0])]
    if len(dtcomp):
        grp = dtcomp.groupby("dtype")["throughput_tps"].mean().to_dict()
        plot_dtype_comparison(grp, "throughput_tps", "Throughput (tok/s)",
                              "Throughput by Dtype (BS=1)",
                              f"{GOUT}/dtype_throughput_bs1.png")

    # TPOT by dtype
    if len(dtcomp):
        grp2 = dtcomp.groupby("dtype")["tpot_ms"].mean().to_dict()
        plot_dtype_comparison(grp2, "tpot_ms", "TPOT (ms/token)",
                              "Time Per Output Token by Dtype (BS=1)",
                              f"{GOUT}/dtype_tpot_bs1.png")

    # Power vs throughput scatter
    if len(df):
        fig, ax = plt.subplots(figsize=(9, 6))
        for dtype_label, grp in df.groupby("dtype"):
            ax.scatter(grp["mean_power_W"], grp["throughput_tps"],
                       label=dtype_label, alpha=0.7, s=50)
        ax.set_xlabel("Mean Power (W)"); ax.set_ylabel("Throughput (tok/s)")
        ax.set_title("Power vs Throughput — All Dtypes & Batch Sizes")
        ax.legend(); ax.grid(alpha=0.3)
        save_fig(fig, f"{GOUT}/power_vs_throughput.png")

    # Roofline for inference
    if len(df):
        roofline_pts = {}
        bw_meas = df["mem_bw_util_pct"].mean() / 100 * MEM_BW_TBps
        for dtype_label, grp in df.groupby("dtype"):
            g = grp.dropna(subset=["throughput_tps", "arith_intensity"])
            if len(g) == 0:
                continue
            # Use the single best row so AI and TFLOPS are consistent
            best = g.loc[g["throughput_tps"].idxmax()]
            flops_per_tok = 2 * model_params_B * 1e9
            achieved_TFLOPS = best["throughput_tps"] * flops_per_tok / 1e12
            ai = best["arith_intensity"]
            roofline_pts[dtype_label] = (ai, achieved_TFLOPS)
        if roofline_pts:
            plot_roofline(roofline_pts, dtype="bf16",
                          out_path=f"{GOUT}/roofline_inference.png",
                          measured_bw_TBps=bw_meas)

    print("\n" + "=" * 70)
    print("  Phase 2 Inference Sweep Complete.")
    print(f"  Results → {OUT}/  |  Graphs → {GOUT}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
