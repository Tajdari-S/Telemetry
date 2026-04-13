"""
Roofline model utilities for NVIDIA B200.
Stores measured and theoretical peak values,
computes arithmetic intensity, and generates roofline plots.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── B200 theoretical specs ──────────────────────────────────────────────────
B200_SPECS = {
    # Memory
    "hbm_bw_TBps":         8.0,        # HBM3e, ~8 TB/s peak
    "hbm_capacity_GB":     183.0,
    "l2_size_MB":          50.0,
    "l1_per_sm_KB":        256.0,
    "num_sms":             160,

    # Compute (TFLOPS, dense)
    "fp64_TFLOPS":         40.0,
    "fp32_TFLOPS":         80.0,
    "tf32_TFLOPS":         2500.0,
    "bf16_TC_TFLOPS":      2250.0,
    "fp16_TC_TFLOPS":      2250.0,
    "fp8_TC_TFLOPS":       4500.0,
    "int8_TC_TOPS":        4500.0,
    "int4_TC_TOPS":        9000.0,

    # NVLink
    "nvlink_links":        18,
    "nvlink_per_link_GBps": 50.0,      # bidirectional per link
    "nvlink_total_GBps":   900.0,      # 18 * 50

    # Power
    "tdp_W":               1000.0,
    "boost_clock_GHz":     1.53,       # approx
}

DTYPE_PEAK = {
    "fp32":    B200_SPECS["fp32_TFLOPS"],
    "tf32":    B200_SPECS["tf32_TFLOPS"],
    "fp16":    B200_SPECS["fp16_TC_TFLOPS"],
    "bf16":    B200_SPECS["bf16_TC_TFLOPS"],
    "fp8":     B200_SPECS["fp8_TC_TFLOPS"],
    "int8":    B200_SPECS["int8_TC_TOPS"],
    "int4":    B200_SPECS["int4_TC_TOPS"],
}

MEM_BW_TBps = B200_SPECS["hbm_bw_TBps"]  # TB/s = TFLOPS when AI=1

def arithmetic_intensity(flops: float, bytes_accessed: float) -> float:
    """FLOP/byte"""
    return flops / bytes_accessed

def roofline_perf(ai: float, peak_flops_T: float, bw_TBps: float) -> float:
    """Attainable TFLOPS given arithmetic intensity."""
    return min(ai * bw_TBps, peak_flops_T)

def ridge_point(peak_flops_T: float, bw_TBps: float) -> float:
    """Ridge point arithmetic intensity (FLOP/byte)."""
    return peak_flops_T / bw_TBps

def plot_roofline(points: dict, dtype: str = "bf16", out_path: str = "roofline.png",
                  measured_bw_TBps: float = None):
    """
    points: dict of label -> (arithmetic_intensity, measured_TFLOPS)
    """
    peak   = DTYPE_PEAK.get(dtype, DTYPE_PEAK["bf16"])
    bw     = measured_bw_TBps if measured_bw_TBps else MEM_BW_TBps
    ridge  = ridge_point(peak, bw)

    ai_range = np.logspace(-3, 4, 500)
    roof_vals = np.minimum(ai_range * bw, peak)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai_range, roof_vals, "b-", lw=2.5, label=f"Roofline ({dtype.upper()})")
    ax.axvline(ridge, color="b", ls="--", lw=1, alpha=0.5)
    ax.text(ridge * 1.05, peak * 0.6,
            f"Ridge={ridge:.1f} FLOP/B", color="b", fontsize=9)

    if measured_bw_TBps and measured_bw_TBps < MEM_BW_TBps:
        roof_meas = np.minimum(ai_range * measured_bw_TBps, peak)
        ax.loglog(ai_range, roof_meas, "g--", lw=1.5,
                  label=f"Measured BW ({measured_bw_TBps:.2f} TB/s)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(points)))
    for (label, (ai, perf)), col in zip(points.items(), colors):
        ax.scatter([ai], [perf], color=col, zorder=5, s=80)
        pct = 100.0 * perf / roofline_perf(ai, peak, bw)
        ax.annotate(f"{label}\n({pct:.0f}%)", (ai, perf),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=col)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(f"Roofline Model — NVIDIA B200 ({dtype.upper()})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Roofline plot saved → {out_path}")
