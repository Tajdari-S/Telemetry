"""Common plotting helpers for all benchmark phases."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import os


PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]

def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_sweep_heatmap(df: pd.DataFrame, x_col: str, y_col: str, val_col: str,
                       title: str, out_path: str, fmt=".1f"):
    pivot = df.pivot(index=y_col, columns=x_col, values=val_col)
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)), max(4, len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=val_col)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        color="white" if val < pivot.values.max() * 0.6 else "black",
                        fontsize=8)
    save_fig(fig, out_path)


def plot_bar_comparison(labels, values, ylabel: str, title: str, out_path: str,
                        color=None, yerr=None):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)*1.2), 5))
    x = range(len(labels))
    bars = ax.bar(x, values, color=color or PALETTE[:len(labels)],
                  yerr=yerr, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, out_path)


def plot_time_series(df: pd.DataFrame, t_col: str, val_cols: list, labels: list,
                     ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    t0 = df[t_col].iloc[0]
    for col, lbl, clr in zip(val_cols, labels, PALETTE):
        ax.plot(df[t_col] - t0, df[col], label=lbl, color=clr, lw=1.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    save_fig(fig, out_path)


def plot_telemetry_dashboard(df: pd.DataFrame, gpu_idx: int, out_path: str):
    cols_labels = [
        ("gpu_util",    "GPU Util (%)"),
        ("mem_util",    "Mem Util (%)"),
        ("power_w",     "Power (W)"),
        ("temp_c",      "Temp (°C)"),
        ("sm_clock_mhz","SM Clock (MHz)"),
        ("mem_clock_mhz","Mem Clock (MHz)"),
    ]
    df_g = df[df["gpu_idx"] == gpu_idx].copy()
    df_g["t"] = df_g["ts"] - df_g["ts"].iloc[0]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle(f"Tier-1 Telemetry Dashboard — GPU {gpu_idx} (NVIDIA B200)", fontsize=13)
    for ax, (col, lbl) in zip(axes.flat, cols_labels):
        ax.plot(df_g["t"], df_g[col], lw=1.2, color=PALETTE[0])
        ax.set_xlabel("Time (s)"); ax.set_ylabel(lbl)
        ax.set_title(lbl); ax.grid(alpha=0.3)
        mn, mx = df_g[col].min(), df_g[col].max()
        ax.axhline(df_g[col].mean(), ls="--", color="red", lw=1, label=f"mean={df_g[col].mean():.1f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, out_path)


def plot_dtype_comparison(results: dict, metric: str, ylabel: str,
                          title: str, out_path: str):
    """results = {dtype_label: metric_value}"""
    labels = list(results.keys())
    values = [results[k] for k in labels]
    colors = PALETTE[:len(labels)]
    plot_bar_comparison(labels, values, ylabel, title, out_path, color=colors)
