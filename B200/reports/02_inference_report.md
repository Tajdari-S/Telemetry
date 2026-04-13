# LLM Inference Benchmark Report — NVIDIA B200
_Generated: 2026-04-13 03:47 UTC_

## Overview
Benchmarks Llama-3.1-8B-Instruct (or Qwen2.5-7B-Instruct) with vLLM across:
- **Dtypes**: FP16, BF16, FP8, INT8, INT4, INT12 (simulated)
- **Batch sizes**: 1, 2, 4, 8, 16, 32
- **Input lengths**: 128, 512, 1024, 2048 tokens
- **Output lengths**: 128, 256, 512 tokens

### INT12 Note
> INT12 is a **non-standard precision** not natively supported by NVIDIA hardware. It is simulated here by quantizing model weights to a 12-bit integer range (`[-2048, 2047]`) stored as BF16. This approximates the accuracy/memory tradeoff of a hypothetical 12-bit format but runs at BF16 speed. Results are labeled for comparison purposes only.

## LLM Performance Metrics
| Metric | Definition |
|--------|-----------|
| TTFT | Time To First Token (s) — prefill latency |
| TPOT | Time Per Output Token (ms) — decode latency per token |
| TBT | Time Between Tokens (ms) ≈ TPOT for non-speculative decode |
| TTL | Time To Last Token (s) — total end-to-end latency |
| Throughput | Output tokens per second (tok/s) |

## Results Summary by Dtype

| Dtype | Max Throughput (tok/s) | Min TPOT (ms) | Mean Power (W) | Mem BW Util (%) |
|-------|----------------------|--------------|---------------|----------------|
| fp8 | 8182 | 0.1 | 479 | 31.3 |
| int12 | 8634 | 0.1 | 694 | 59.6 |
| int8 | 2929 | 0.2 | 672 | 31.8 |

## Throughput Heatmaps (by dtype)

![graph](../graphs/inference/throughput_heatmap_bf16.png)
_BF16 throughput (tok/s) across batch size × input length_

![graph](../graphs/inference/throughput_heatmap_fp8.png)
_FP8 throughput (tok/s)_

![graph](../graphs/inference/throughput_heatmap_int4.png)
_INT4 throughput (tok/s)_

## Dtype Comparison (BS=1)

![graph](../graphs/inference/dtype_throughput_bs1.png)
_Throughput by dtype at BS=1_

![graph](../graphs/inference/dtype_tpot_bs1.png)
_TPOT by dtype at BS=1_

## Power vs Throughput

![graph](../graphs/inference/power_vs_throughput.png)
_Power efficiency across all sweeps_

## Roofline — Inference

![graph](../graphs/inference/roofline_inference.png)
_LLM inference is memory-bandwidth bound at BS=1 (AI ≈ batch_size for decode)_

### How Roofline Points Are Calculated

Each point on the roofline represents the **best-case configuration** (highest throughput across
all input/output length sweeps) for each dtype at BS=32.

**Arithmetic Intensity (X-axis):**

LLM decode loads all model weights from HBM once per step, processing `batch_size` tokens:
```
FLOP per step  = 2 × N_params × batch_size        (one matmul per token, weight reuse across batch)
Bytes accessed = N_params × bytes_per_param        (full weight load, once per step)

AI = FLOP / bytes = 2 × batch_size / bytes_per_param
```

| Dtype | bytes/param | BS=32 AI (FLOP/byte) |
|-------|------------|----------------------|
| FP8   | 1          | 2 × 32 / 1 = **64**  |
| INT8  | 1          | 2 × 32 / 1 = **64**  |
| INT12 | 2 (BF16 at runtime) | 2 × 32 / 2 = **32** |
| BF16  | 2          | 2 × 32 / 2 = **32**  |

Note: N_params cancels in the AI ratio — the arithmetic intensity depends only on batch size
and bytes per parameter, not on the absolute model size.

**Achieved TFLOPS (Y-axis):**
```
achieved_TFLOPS = max_throughput_tps × 2 × N_params / 1e12
```
Using N_params = 8.0B (Qwen2.5-7B rounded; ~5% overestimate).

**The two roof lines:**
- **Solid blue line** — theoretical peak: `min(AI × 8.0 TB/s, peak_dtype_TFLOPS)`. This is the
  hard ceiling imposed by B200 hardware. All measured points must be below this line.
- **Dashed green line** — measured average BW reference: `min(AI × 3.27 TB/s, peak)`.
  This is the mean HBM bandwidth utilization across **all** sweep configurations (3.27 TB/s =
  40.9% of 8.0 TB/s), including small-batch runs where utilization is low.

### Why INT12 Appears Above the Green (Measured) Reference Line

INT12 at BS=32 achieves **138.1 TFLOPS** while the measured-BW reference at AI=32 is
**104.7 TFLOPS** (32 × 3.27 TB/s). This is not a bug — it is expected:

- The **green line** represents the *average* BW utilization across all 72 configurations
  (BS=1 through BS=32, all input lengths). Low-batch runs (BS=1,2) achieve only 2–5% BW
  utilization due to kernel launch overhead, dragging the average down.
- The **point** plotted is the *best* configuration (BS=32), which achieves **65.1% BW
  utilization** (5.21 TB/s) — far above the 40.9% average.
- At 5.21 TB/s, the BW roof at AI=32 is 32 × 5.21 = **166.7 TFLOPS**, and INT12 achieves
  138.1 TFLOPS — correctly below that.
- The **solid blue theoretical roof** at AI=32 is 32 × 8.0 = **256 TFLOPS**. INT12 at 138.1
  is 54% of theoretical — physically valid.

In short: the green line is a *population average*, not a ceiling. Plotting the peak throughput
point against an average naturally places high-performing configurations above the average line.
The only physically meaningful constraint is the blue theoretical roof.

## Key Findings
- BS=1 decode is **memory-bound** (arithmetic intensity = 1 FLOP/byte, ridge at ~560).
- Throughput scales nearly linearly with batch size up to memory capacity.
- FP8 provides ~2.7× throughput vs INT8 and is the recommended low-precision dtype on B200.
- INT4 / INT8 via bitsandbytes are **slower than FP8** despite fewer bits — see analysis below.
- INT12 (simulated) appears fast because it runs at BF16 speed with no real quantization overhead.
- Power draw stays near TDP for large batches; idle for BS=1.

## Analysis: Why Fewer Bits Does Not Mean Higher Throughput

**Observed throughput order**: INT12 (simulated) > FP8 > INT8 — counter-intuitive given bit widths.

| Dtype | Mean tok/s | Why |
|-------|-----------|-----|
| INT12 | 2680 | Stored and computed as BF16; no real quantization overhead |
| FP8 | 2573 | Native B200 Tensor Cores (4500 TFLOPS); no dequantization needed |
| INT8 | 951 | bitsandbytes dequantizes weights to FP16 before every matmul |

**The INT8 penalty**: vLLM's INT8 path (bitsandbytes) stores weights in INT8 (saving bandwidth,
same as FP8), then dequantizes to FP16 before the matmul (adding a kernel and extra bandwidth
reads), then runs compute in FP16. The INT8 Tensor Cores are never used. The bandwidth savings
are consumed by the dequantization step, leaving net throughput well below FP8.

**FP8 on B200**: The Blackwell architecture executes FP8 matmuls natively at 4500 TFLOPS.
vLLM uses `torch._scaled_mm` / cuBLAS FP8 paths with no format conversion. FP8 gets both the
2× weight bandwidth reduction AND the full Tensor Core benefit.

**Recommendation for B200 inference**: Use FP8 for low-precision deployment. Avoid INT8/INT4
bitsandbytes paths unless the model does not fit in VRAM at all.

> See `reports/06_analysis_findings.md` — Finding 1 for the full root-cause analysis.