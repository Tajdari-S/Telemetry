# Analysis & Findings — NVIDIA B200 Benchmark Suite
_Last updated: 2026-04-13_

This report documents key observations, counter-intuitive results, and root-cause analyses
that emerged from running the full benchmark suite. Each finding answers a specific question
raised during post-run analysis.

---

## Finding 1 — Inference Throughput Drops When Bit-Width Decreases

### Question
Why does int8 inference produce lower throughput than fp8, even though int8 has fewer bits?

### Observed Data

| Dtype | Mean Throughput (tok/s) | Mean Mem BW Util (%) | Mean Power (W) |
|-------|------------------------|---------------------|----------------|
| int12 (simulated) | 2680 | 59.6 | 694 |
| fp8 | 2573 | 31.3 | 479 |
| int8 (bitsandbytes) | 951 | 31.8 | 672 |

int8 is **2.7× slower** than fp8 despite representing smaller weights.

### Root Cause: Hardware-Native vs Software-Emulated Quantization

The B200 Blackwell architecture has **native Tensor Core support for FP8**:
- FP8 matmuls execute directly at 4500 TFLOPS peak
- vLLM uses `torch._scaled_mm` / cuBLAS FP8 paths with no format conversion
- Weight memory is halved (vs BF16) AND compute runs on the fastest path

**INT8 via bitsandbytes follows a different (slower) path:**
1. Weights are stored in INT8 — saves memory bandwidth vs BF16 (same gain as FP8)
2. Before each matmul, weights are **dequantized back to FP16/BF16** — adds a full extra kernel
3. The actual matmul runs in FP16/BF16, so no INT8 Tensor Core benefit is captured
4. Net effect: bandwidth savings are consumed by the dequantization step

The result is a workload that gets the worst of both worlds: the complexity of quantization
with none of the compute speedup.

**INT12** appears fastest because it is not real quantization — weights are clamped to a 12-bit
integer range but stored and computed in BF16. It runs at full BF16 speed with no overhead.

### Rule of Thumb for B200 Inference

| Path | Inference Speed | Memory | When to Use |
|------|----------------|--------|-------------|
| BF16 | Baseline | 2 bytes/param | Max quality, fits in VRAM |
| FP8 (native) | ~1.7× BF16 | 1 byte/param | Production inference on Blackwell |
| INT8 (bitsandbytes) | ~0.4× BF16 | 1 byte/param | Avoid unless VRAM is the only constraint |
| INT4 (bitsandbytes) | <INT8 in practice | 0.5 bytes/param | Only when model does not fit otherwise |

For B200, **FP8 is the recommended low-precision inference dtype**. INT8/INT4 via bitsandbytes
were designed for older architectures and add dequantization overhead that negates their benefit
on Blackwell.

---

## Finding 2 — Training MFU Exceeds 100% (Calibration Error)

### Question
Why does the training roofline show workloads above the memory saturation line, and why is MFU > 100%?

### Observed Data (bs=4, seq=512)

| Dtype | Tokens/s | Reported MFU (%) | Note |
|-------|----------|-----------------|------|
| bf16 | 46258 | 393% | Impossible — exceeds 100% |
| int12 | 45089 | 383% | Impossible |
| int8 | 46162 | 196% | Impossible |
| int4 | 44623 | 95% | Plausible, possibly correct by accident |
| fp32 | 4897 | 1171% | Impossible — 12× over peak |

### Root Cause: Two Compounding Bugs in the MFU Formula

**Bug 1 — Arithmetic intensity formula is wrong.**
The script calculated `arith_intensity` as a simple ratio of dtype byte widths (e.g., 1.5 for fp32,
3.0 for bf16). The correct formula for a transformer forward+backward pass is:

```
FLOP per step = 6 × N_params × batch_size × seq_len
                ↑ factor of 6: ≈2 for forward, ≈4 for backward (gradients + weight update)

Bytes loaded   = N_params × bytes_per_dtype   (weight loads dominate at large batch)

Arithmetic Intensity (FLOP/byte) = FLOP / Bytes
                                 = 6 × batch_size × seq_len / bytes_per_dtype_ratio
```

At bs=4, seq=512, bf16 (2 bytes/param):
```
AI = 6 × 4 × 512 / 2 = 6144 FLOP/byte
```

This places training **far above the B200 ridge point** (281 FLOP/byte for BF16), meaning
training is deeply compute-bound — which is correct. The script's value of 3.0 is off by ~2000×.

**Bug 2 — MFU numerator and denominator are inconsistent.**
MFU should be:
```
MFU = achieved_TFLOPS / peak_TFLOPS
    = (tokens_per_sec × FLOP_per_token) / peak_TFLOPS_for_dtype
```

The script used a `scaled_tps` approach that multiplied throughput by a factor to pretend the
proxy model was 8B parameters, but then divided by the wrong peak TFLOPS, producing values
that exceed 1.0 (100%).

### Corrected Interpretation

Training in all dtypes runs in the **compute-bound region** of the roofline (AI >> 281 FLOP/byte
for BF16, AI >> 562 for FP8). The roofline plots with the wrong AI values misleadingly show
workloads in the memory-bound region. The throughput numbers (tokens/sec) are real and valid;
only the roofline position and MFU percentages are affected by this calibration error.

**Practical implication:** training throughput being nearly equal across BF16, INT8, INT4, and
INT12 is real and expected — when compute-bound, dtype only affects compute-ceiling height, not
memory bandwidth, so the proxy model's small parameter count means all non-fp32 dtypes hit the
same ceiling.

---

## Finding 3 — 2-GPU Per-GPU Power Is Lower Than 1-GPU Power

### Question
Why does each GPU use less power in 2-GPU mode than in 1-GPU mode?

### Observed Data (SPAR workloads, `mean_power_W` = per-GPU average)

| Workload | 1× GPU (W/GPU) | 2× GPU (W/GPU each) | Total 2× (W) | GPU Util 1× (%) | GPU Util 2× (%) |
|----------|---------------|---------------------|--------------|----------------|----------------|
| scientific_hpc | 797 | 495 | **990** | 98.8 | 49.3 |
| modern_inference | 665 | 430 | **860** | 94.8 | 47.2 |
| cufft_benchmark | 386 | 287 | **574** | 93.8 | 46.8 |
| nbody_sim | 526 | 358 | **716** | 93.6 | 46.9 |
| resnet50_inference | 469 | 252 | **504** | 90.0 | 5.8 |
| pytorch_training | 318 | 250 | **500** | 60.9 | 9.3 |

### Root Cause: Three Factors Acting Together

**1. The metric is per-GPU, not total system power.**
`mean_power_W` records the average power of a single GPU. In 2-GPU mode, the workload is split
across both devices. To compare total energy draw, multiply per-GPU power by the number of GPUs.
Total 2× power is always higher than 1× power.

**2. Splitting work halves per-GPU utilization.**
For workloads that parallelize well (scientific_hpc, cufft, nbody), GPU utilization in 2× mode
is almost exactly half of 1× mode (49% vs 99%, 47% vs 94%). Each GPU does half the work.

**3. Power scales sublinearly with GPU utilization.**
GPU power is not proportional to utilization. There is a fixed idle baseline (~145 W/GPU for B200
in persistence mode) plus a variable active component:

```
Power ≈ P_idle + (P_TDP − P_idle) × utilization

At 95% util: 145 + (1000 − 145) × 0.95 ≈ 957 W
At 47% util: 145 + (1000 − 145) × 0.47 ≈ 547 W
```

Halving utilization does not halve power — it only reduces the active portion, so per-GPU power
drops by ~410 W while total system power still rises.

**4. Special case — resnet50 and pytorch_training (DataParallel overhead).**
GPU utilization in 2× mode drops to 5.8% (resnet50) and 9.3% (pytorch_training). DataParallel
places the entire model on both GPUs, splits the batch, runs forward/backward on each, then
synchronizes gradients via all-reduce. For small models and small batches, synchronization time
dominates: GPUs spend most of the interval idle, waiting for each other. Both GPUs consumed power
at near-idle rates while producing almost no useful work.

### Summary

| Observation | Explanation |
|-------------|-------------|
| Per-GPU power is lower in 2× mode | Each GPU runs at half utilization (work split) |
| Total system power is always higher in 2× mode | You have two GPUs, not one |
| Power drops less than utilization | Fixed idle baseline (~145 W) is constant |
| Some workloads show very low 2× GPU util | DataParallel synchronization overhead dominates |

### Recommended Metric: Performance per Watt

Raw power draw is misleading for comparing 1× and 2× configurations. Use `perf_per_watt`
(iterations per second per watt of total system power):

```
perf_per_watt = iters / (num_gpus × mean_power_W_per_gpu)
```

Workloads that parallelize well (scientific_hpc, nbody, cufft) show near-constant or improving
perf/watt with 2 GPUs. Workloads dominated by synchronization overhead (resnet50, pytorch_training
with small batch) show degraded perf/watt and should not be run in DataParallel mode at small scale.

---

---

## Finding 4 — Roofline Methodology and Why a Point Can Be Above the Measured BW Line

### Question
How are the roofline points calculated, and why does INT12 appear above the measured bandwidth reference line?

### How Arithmetic Intensity Is Calculated (Inference Decode)

During the decode phase of LLM inference, each step loads all model weights from HBM once,
processes `batch_size` tokens, and produces one output token per sequence:

```
FLOP per step  = 2 × N_params × batch_size
                 ↑ factor 2: multiply-accumulate in each linear layer

Bytes accessed = N_params × bytes_per_param
                 ↑ weight load dominates; activations (M×K) are negligible vs weights (N×K) at small batch

Arithmetic Intensity = FLOP / bytes = 2 × batch_size / bytes_per_param
```

N_params cancels in the ratio — AI depends only on batch size and bytes per parameter,
not on the absolute model size.

| Dtype | bytes/param at runtime | BS=32 AI (FLOP/byte) |
|-------|------------------------|----------------------|
| FP8   | 1                      | 2 × 32 / 1 = **64**  |
| INT8 (bitsandbytes) | 1 (stored) → 2 (after dequant) | 2 × 32 / 1 = **64** |
| INT12 | 2 (BF16 at runtime)    | 2 × 32 / 2 = **32**  |
| BF16  | 2                      | 2 × 32 / 2 = **32**  |

### How Achieved TFLOPS Is Calculated

```
achieved_TFLOPS = throughput_tps × 2 × N_params / 1e12
```

The `throughput_tps` is the peak measured output tokens per second across all input/output
length combinations at the given batch size. N_params = 8.0B (Qwen2.5-7B, ~5% overestimate).

### The Two Roof Lines

| Line | Formula | What it represents |
|------|---------|-------------------|
| Solid blue (theoretical) | `min(AI × 8.0 TB/s, peak_TFLOPS)` | Hard hardware ceiling — no measurement can exceed this |
| Dashed green (measured avg) | `min(AI × 3.27 TB/s, peak_TFLOPS)` | Mean HBM BW utilization across ALL 72 sweep configs (40.9% × 8.0 TB/s) |

### Why INT12 Appears Above the Green Reference Line

INT12 at BS=32 achieves **138.1 TFLOPS** while the measured-BW reference at AI=32 gives
**32 × 3.27 = 104.7 TFLOPS**. This is expected behavior, not a bug:

- The **green line** is a *population average* over all 72 configurations. Small-batch runs
  (BS=1,2) achieve only 2–5% BW utilization due to kernel overhead, pulling the mean down to 40.9%.
- The **plotted point** is the *best case* (BS=32), which achieves **65.1% BW utilization**
  (5.21 TB/s) — well above the average.
- At 5.21 TB/s, the BW roof at AI=32 is 32 × 5.21 = **166.7 TFLOPS**. INT12 at 138.1 is below it.
- The **theoretical roof** at AI=32 is 32 × 8.0 = **256 TFLOPS**. INT12 at 138.1 is 54% of that.

Plotting the peak-throughput point against a population-average reference line will naturally
place high-performing configurations above the reference. The only binding physical constraint
is the blue theoretical roof.

### Summary of Numbers

| Dtype | Best BS | AI (FLOP/B) | Achieved TFLOPS | Theory roof | Measured BW ref | Above ref? |
|-------|---------|------------|-----------------|-------------|-----------------|-----------|
| FP8   | 32 | 64 | 130.9 | 512 | 209.4 | No |
| INT12 | 32 | 32 | 138.1 | 256 | 104.7 | **Yes** (expected) |
| INT8  | 32 | 64 |  46.9 | 512 | 209.4 | No |

---

## Finding 5 — Dequantization: Memory-Bound But Not HBM-Saturating

### Question
Is dequantization memory-bound or compute-bound? Why is HBM BW utilization low even for large tensors?

### Arithmetic Intensity of the Dequant Kernel

The dequantization step (`INT8 → BF16`) for a weight matrix of N×K elements:
```
Bytes read    = N × K × 1    (INT8, 1 byte each)
Bytes written = N × K × 2    (BF16, 2 bytes each)
FLOP          = N × K × 1    (one multiply per element: val × scale)

AI = FLOP / (bytes_read + bytes_written) = N×K / (N×K × 3) = 1/3 ≈ 0.33 FLOP/byte
```

Ridge point on B200 (BF16): **281 FLOP/byte**. Dequantization at 0.33 is **850× below** the
ridge — it is always deeply memory-bound regardless of tensor size.

### Measured Dequant Kernel Performance (B200, INT8→BF16)

| Weight matrix | Size (MB) | Actual latency | Ideal (8 TB/s) | BW util |
|--------------|----------|---------------|----------------|---------|
| 256×512      | 0.1      | 0.019 ms      | 0.000 ms       | 0.3%    |
| 1024×1024    | 1        | 0.022 ms      | 0.000 ms       | 1.8%    |
| 2048×4096    | 8        | 0.047 ms      | 0.003 ms       | 6.6%    |
| 4096×14336   | 59       | 0.341 ms      | 0.022 ms       | 6.5%    |
| 8192×14336   | 117      | 0.668 ms      | 0.044 ms       | 6.6%    |
| 8192×28672   | 235      | 1.324 ms      | 0.088 ms       | 6.7%    |
| 16384×28672  | 470      | 2.637 ms      | 0.176 ms       | 6.7%    |

### Why Only 6.7% of HBM Bandwidth?

Two distinct regimes:

**Small tensors (<8 MB): kernel launch overhead dominates.**
The CUDA kernel launch latency (~10–20 µs) exceeds the actual transfer time. A 1 MB tensor
should take ~0.4 µs at 8 TB/s but the kernel costs ~22 µs total → ~2% BW utilization.

**Large tensors (59–470 MB): kernel inefficiency dominates.**
Latency scales linearly with size (confirming it is memory-bound, not compute-bound), but
BW utilization plateaus at **~6.7%** rather than approaching 100%. The cause is that
`W.to(torch.bfloat16) * scale` in PyTorch is not a fused kernel on B200 — it issues two
separate element-wise passes (cast pass + multiply pass), neither of which uses the B200's
full 8 TB/s HBM bandwidth. Actual achieved bandwidth: ~535 GB/s (~6.7% of 8 TB/s).

### Consequence for INT8 Inference

The dequant kernel adds **~0.34 ms of fixed overhead** per linear layer per forward pass,
regardless of batch size. At BS=1 where the BF16 matmul takes only 0.026 ms, the dequant step
is **13× longer than the actual matmul**. Even at BS=2048, dequant (0.34 ms) consumes ~25%
of the total forward pass budget.

This is why INT8 via bitsandbytes is slower than native FP8: the dequant overhead is a
constant tax per layer that dwarfs the bandwidth savings from halving weight size.

---

## Summary Table

| Finding | Short Answer |
|---------|-------------|
| Why does int8 inference underperform fp8? | fp8 runs natively on B200 Tensor Cores; int8 (bitsandbytes) dequantizes to fp16 before compute, adding overhead that cancels the memory savings |
| Why is training MFU > 100%? | Arithmetic intensity and MFU were computed with wrong formulas. Real AI is ~6×bs×seq/bytes_per_param (thousands of FLOP/byte); training is deeply compute-bound |
| Why is 2-GPU per-GPU power lower than 1-GPU? | The metric is per-GPU; work splits across two GPUs, halving per-GPU utilization. Total system power is always higher. Power vs utilization is sublinear due to a fixed ~145 W idle floor |
| Why is INT12 above the measured BW reference line? | The green line is a population average (40.9% BW util) across all batch sizes; the INT12 point is the best case (BS=32, 65.1% BW util). Best-case points naturally exceed population-average lines. Only the theoretical roof is a physical bound |
| Is dequantization memory-bound or compute-bound? | Memory-bound (AI = 0.33 FLOP/byte, 850× below the B200 ridge). But PyTorch's non-fused cast+multiply only achieves ~535 GB/s (6.7% of 8 TB/s HBM peak) — adding ~0.34 ms fixed overhead per layer regardless of batch size |
