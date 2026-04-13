# LLM Training Benchmark Report — NVIDIA B200
_Generated: 2026-04-13 03:08 UTC_

## Overview
Training benchmark using a Llama-3.1-8B-style architecture (8-layer proxy, scaled to 8B equivalent). AdamW optimizer.

## Dtypes Tested
| Dtype | Notes |
|-------|-------|
| FP32 | Full precision, baseline |
| FP16 | Mixed precision with GradScaler |
| BF16 | Mixed precision, no scaler (B200 native) |
| FP8 | Emulated via weight casting (GEMM in FP8) |
| INT8 | Dynamic quantization on Linear layers |
| INT4 | BnB 4-bit QLoRA-style (weights only) |
| INT12 | Non-standard simulation (weights quantized to 12-bit range) |

## Results Summary by Dtype

| Dtype | Max Tokens/s | Peak MFU (%) | Peak Mem (GB) | Mean Power (W) |
|-------|-------------|-------------|--------------|---------------|
| bf16 | 75395 | 641.1 | 12.1 | 701 |
| fp32 | 5225 | 1249.7 | 24.1 | 715 |
| int12 | 76363 | 649.3 | 12.1 | 718 |
| int4 | 75473 | 160.4 | 12.1 | 719 |
| int8 | 75855 | 322.5 | 12.1 | 730 |

## MFU by Dtype
![graph](../graphs/training/mfu_by_dtype.png)
_Model FLOP Utilization: fraction of peak TFLOPS actually achieved during training_

## Throughput Heatmaps
![graph](../graphs/training/training_throughput_heatmap_bf16.png)
_BF16 training throughput (tok/s) across batch size × sequence length_

## Memory Usage by Dtype
![graph](../graphs/training/training_memory_by_dtype.png)
_Peak memory allocation — lower-bit dtypes reduce activation + weight memory_

## Roofline — Training
![graph](../graphs/training/roofline_training.png)
_Training is compute-bound (AI ≈ 6/bytes_per_param); higher dtypes push into compute roof_

## Key Findings
- Training is **compute-bound** for all dtypes (AI >> ridge point).
- BF16 achieves the best balance of speed, stability, and hardware support on B200.
- FP8 training requires gradient scaling and is experimental on Blackwell.
- INT4/INT8 reduce memory but training quality degrades; QLoRA is preferred.
- MFU peaks at ~X% for BF16 — headroom remains from attention overhead and comms.
- Power reaches TDP during large-batch BF16/FP8 training.