# NVIDIA B200 Benchmark Suite

Full-system characterization and LLM performance profiling for **NVIDIA B200** (Blackwell).

## Hardware Tested
| | GPU 0 | GPU 1 |
|---|---|---|
| **Model** | NVIDIA B200 | NVIDIA B200 |
| **Memory** | 183 GB HBM3e | 183 GB HBM3e |
| **Interconnect** | NVLink18 (18 × 50 GB/s = **900 GB/s**) | |
| **TDP** | 1000 W | 1000 W |
| **CUDA** | 12.8 | |

## Benchmark Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `scripts/01_system_profile.py` | Memory BW, cache hierarchy, TLB, SM/TC GEMM, roofline |
| 2 | `scripts/02_tier1_telemetry.py` | NVML Tier-1 telemetry across workloads |
| 3 | `scripts/03_llm_inference_sweep.py` | LLM inference: all dtypes × batch/prompt sweeps |
| 4 | `scripts/04_llm_training_sweep.py` | LLM training: same sweeps + MFU |
| 5 | `scripts/05_nvlink_comparison.py` | NVLink vs 1x GPU: P2P BW, TP-GEMM, all-reduce |
| 6 | `scripts/06_generate_reports.py` | Auto-generates Markdown reports from results |

## Quick Start

```bash
# Run all phases (auto-selects open model if no HF token)
bash scripts/run_all.sh

# With Llama 3.1 (requires HF token + Meta access agreement)
bash scripts/run_all.sh --hf-token hf_YOUR_TOKEN

# Quick test run (reduced sweeps)
bash scripts/run_all.sh --quick

# Skip inference (for training + telemetry only)
bash scripts/run_all.sh --skip-inference
```

## Dtype Coverage

| Dtype | HW Support | Notes |
|-------|-----------|-------|
| FP32 | Native | Baseline, ~80 TFLOPS |
| TF32 | Native TC | ~2500 TFLOPS (auto with `allow_tf32`) |
| BF16 | Native TC | ~2250 TFLOPS, B200 preferred training dtype |
| FP16 | Native TC | ~2250 TFLOPS, requires GradScaler |
| FP8 (E4M3) | Native TC | ~4500 TFLOPS, Blackwell native |
| INT8 | Native TC | ~4500 TOPS |
| INT4 | Native TC | ~9000 TOPS |
| **INT12** | **Non-standard** | Simulated (no HW support); see reports for details |

## Key B200 Roofline Parameters

```
Peak BF16 TFLOPS:   2250
Peak FP8 TFLOPS:    4500
HBM3e BW:          8.0 TB/s  (theoretical)
L2 Cache:           50 MB
L1 Cache/SM:        256 KB
# SMs:              160
Ridge point (BF16): 2250 / 8.0 = 281 FLOP/byte
Ridge point (FP8):  4500 / 8.0 = 562 FLOP/byte
NVLink18 BW:        900 GB/s (per direction)
```

## Directory Structure

```
B200/
├── scripts/
│   ├── 01_system_profile.py
│   ├── 02_tier1_telemetry.py
│   ├── 03_llm_inference_sweep.py
│   ├── 04_llm_training_sweep.py
│   ├── 05_nvlink_comparison.py
│   ├── 06_generate_reports.py
│   ├── run_all.sh
│   └── utils/
│       ├── telemetry.py      # Tier-1 NVML collector
│       ├── roofline.py       # Roofline model + B200 specs
│       └── plotting.py       # Common plotting helpers
├── results/                  # Raw CSV/JSON data
│   ├── system_profile/
│   ├── inference/
│   ├── training/
│   ├── telemetry/
│   └── nvlink/
├── graphs/                   # PNG figures
└── reports/                  # Markdown reports
    ├── 01_system_profile_report.md
    ├── 02_inference_report.md
    ├── 03_training_report.md
    ├── 04_telemetry_report.md
    └── 05_nvlink_report.md
```

## Tier-1 Telemetry Definition

Tier-1 follows the [SPAR-GPU-monitoring](https://github.com/robirahman/SPAR-GPU-monitoring) protocol:
- **Tool**: `pynvml` (no DCGM required)
- **Rate**: 20 Hz
- **Metrics**: GPU util, mem util, power, temp, SM clock, mem clock, NVLink Rx/Tx, PCIe Rx/Tx, ECC

## References
- SPAR-GPU-monitoring: https://github.com/robirahman/SPAR-GPU-monitoring
- Telemetry repo: https://github.com/Tajdari-S/Telemetry
- NVIDIA B200 datasheet: https://www.nvidia.com/en-us/data-center/b200/
