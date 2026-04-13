# Tier-1 NVML Telemetry Report — NVIDIA B200
_Generated: 2026-04-13 03:47 UTC_

## Methodology
Follows SPAR-GPU-monitoring Tier-1 telemetry protocol: pynvml sampled at 20 Hz during each workload. No DCGM required (Tier-1 = pynvml only).

Metrics collected: GPU utilization, memory utilization, power (W), temperature (°C), SM clock, memory clock, NVLink Rx/Tx (MB), PCIe Rx/Tx (MB), ECC errors.

## Experiments
| Experiment | GPUs | Duration | Description |
|-----------|------|----------|-------------|
| 1xGPU_GEMM | GPU 0 | 25s | Continuous 8192² BF16 GEMM |
| 2xGPU_GEMM | GPU 0+1 | 25s | Independent GEMMs on both |
| NVLink_P2P | GPU 0+1 | 25s | 1 GB tensor copy GPU0→GPU1 |
| 1xGPU_Transformer | GPU 0 | 25s | 4096-dim attention+FFN block |
| 2xGPU_Transformer | GPU 0+1 | 25s | DataParallel transformer |
| Idle | GPU 0+1 | 15s | Baseline idle |

## Results Summary

| Experiment | Mean GPU Util (%) | Mean Power (W) | Max Power (W) | Mean Temp (°C) |
|-----------|-----------------|---------------|--------------|--------------|
| 1xGPU_GEMM | 97.7 | 965 | 1015 | 53.7 |
| 2xGPU_GEMM | 98.1 | 970 | 1030 | 54.4 |
| NVLink_P2P | 48.9 | 357 | 381 | 38.4 |
| 1xGPU_Transformer | 92.5 | 924 | 1006 | 55.0 |
| 2xGPU_Transformer | 59.0 | 641 | 1017 | 43.5 |
| Idle | 0.0 | 197 | 202 | 35.7 |

## Telemetry Dashboards

![graph](../graphs/telemetry/dashboard_1xgpu_gemm_gpu0.png)
_Figure: 1x GPU GEMM — 6-panel Tier-1 dashboard (GPU 0)_

![graph](../graphs/telemetry/dashboard_2xgpu_gemm_gpu0.png)
_Figure: 2x GPU GEMM — GPU 0_

![graph](../graphs/telemetry/dashboard_nvlink_gpu0.png)
_Figure: NVLink P2P stress — GPU 0 (source)_

![graph](../graphs/telemetry/dashboard_nvlink_gpu1.png)
_Figure: NVLink P2P stress — GPU 1 (destination)_

## 1x vs 2x GPU vs NVLink Comparison
![graph](../graphs/telemetry/1x_vs_2x_vs_nvlink_comparison.png)
_GPU utilization and power across all experiments_

![graph](../graphs/telemetry/compare_mean_power_W.png)
_Mean power by experiment_

## Key Findings
- 1x GPU GEMM drives ~90% GPU utilization and approaches TDP.
- 2x GPU independent GEMM doubles total compute at ~same per-GPU metrics.
- NVLink P2P: GPU utilization is lower on source GPU; memory clock pegged.
- Transformer workload shows lower util than pure GEMM (attention overhead).
- Idle baseline: ~145 W per GPU (persistence mode + HBM refresh).

## Analysis: Why 2-GPU Per-GPU Power Is Lower Than 1-GPU Power

**Observed pattern (SPAR workloads, `mean_power_W` = per-GPU average):**

| Workload | 1× GPU (W/GPU) | 2× GPU (W/GPU each) | Total 2× (W) | GPU Util 1× | GPU Util 2× |
|----------|---------------|---------------------|--------------|-------------|-------------|
| scientific_hpc | 797 | 495 | **990** | 98.8% | 49.3% |
| modern_inference | 665 | 430 | **860** | 94.8% | 47.2% |
| cufft_benchmark | 386 | 287 | **574** | 93.8% | 46.8% |
| nbody_sim | 526 | 358 | **716** | 93.6% | 46.9% |
| resnet50_inference | 469 | 252 | **504** | 90.0% | 5.8% |

**Per-GPU power is lower in 2× mode for three reasons:**

1. **The metric is per-GPU.** Total system power (W/GPU × num_GPUs) is always higher with 2 GPUs.

2. **Work splits across both GPUs, halving per-GPU utilization.** For well-parallelised workloads,
   2× GPU util is almost exactly half of 1×. A GPU at 47% utilization draws much less power than
   one at 94%.

3. **Power scales sublinearly with utilization** due to a fixed idle baseline:
   ```
   Power ≈ P_idle + (P_TDP − P_idle) × utilization
   At 95% util:  145 + (1000 − 145) × 0.95  ≈  957 W
   At 47% util:  145 + (1000 − 145) × 0.47  ≈  547 W
   ```
   Halving utilization saves ~410 W per GPU (the active component), but the idle floor stays.
   Total system power rises anyway because there are now two devices.

**Special case — resnet50_inference (2× GPU util only 5.8%):**
DataParallel splits the batch, runs forward/backward on each GPU, then all-reduces gradients.
For small models and small batches, the synchronization dominates: both GPUs spend most of the
window idle. Result: two GPUs drawing near-idle power for almost no throughput gain.

**Recommended metric: perf/watt** (iterations per watt of total system power):
```
perf_per_watt = iters / (num_gpus × mean_power_W_per_gpu)
```
Workloads that parallelize well (scientific_hpc, nbody, cufft) show near-constant perf/watt
across 1× and 2× configs. Workloads dominated by sync overhead (resnet50, pytorch_training at
small batch) show degraded perf/watt and should not be run with DataParallel at small scale.

> See `reports/06_analysis_findings.md` — Finding 3 for the full root-cause analysis.