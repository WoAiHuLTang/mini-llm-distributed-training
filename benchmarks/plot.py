"""Generate benchmark plots from the results CSV.

Produces:
    - memory.png            : peak GPU memory per strategy
    - throughput.png        : tokens/s per strategy
    - scaling_efficiency.png: throughput normalized per GPU (scaling efficiency)

Usage:
    python benchmarks/plot.py --csv benchmarks/results/benchmark.csv \
        --outdir benchmarks/results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _label(row: pd.Series) -> str:
    """Human-readable strategy label, e.g. 'FSDP+AC (2 GPU)'."""
    name = row["strategy"].upper()
    if row["strategy"] == "fsdp" and row["activation_checkpointing"]:
        name = "FSDP+AC"
    return f"{name} ({int(row['gpus'])} GPU)"


def plot_memory(df: pd.DataFrame, outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = df.apply(_label, axis=1)
    colors = plt.cm.viridis(df.index / max(len(df) - 1, 1))
    bars = ax.bar(labels, df["memory_gb"], color=colors)
    for bar, val in zip(bars, df["memory_gb"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Peak GPU Memory by Strategy")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "memory.png"), dpi=150)
    plt.close()


def plot_throughput(df: pd.DataFrame, outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = df.apply(_label, axis=1)
    colors = plt.cm.plasma(df.index / max(len(df) - 1, 1))
    bars = ax.bar(labels, df["tokens_per_s"], color=colors)
    for bar, val in zip(bars, df["tokens_per_s"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput by Strategy")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "throughput.png"), dpi=150)
    plt.close()


def plot_scaling_efficiency(df: pd.DataFrame, outdir: str) -> None:
    """Throughput per GPU, normalized to the single-GPU baseline (100%)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline = single GPU throughput.
    single = df[df["strategy"] == "single"]
    if single.empty:
        print("No single-GPU baseline found; skipping scaling efficiency plot.")
        return
    base_tps = float(single.iloc[0]["tokens_per_s"])

    labels, eff = [], []
    for _, row in df.iterrows():
        per_gpu = row["tokens_per_s"] / row["gpus"]
        eff.append(per_gpu / base_tps * 100.0)
        labels.append(_label(row))

    colors = plt.cm.coolwarm([e / 100.0 for e in eff])
    bars = ax.bar(labels, eff, color=colors)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Single-GPU baseline")
    for bar, val in zip(bars, eff):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Scaling Efficiency per GPU (%)")
    ax.set_title("Scaling Efficiency (throughput/GPU vs single GPU)")
    ax.set_ylim(0, max(max(eff) * 1.15, 110))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scaling_efficiency.png"), dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot benchmark results")
    p.add_argument("--csv", type=str, default="benchmarks/results/benchmark.csv")
    p.add_argument("--outdir", type=str, default="benchmarks/results")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    print("Benchmark summary:")
    print(df.to_string(index=False))

    plot_memory(df, args.outdir)
    plot_throughput(df, args.outdir)
    plot_scaling_efficiency(df, args.outdir)
    print(f"Plots written to {args.outdir}/")


if __name__ == "__main__":
    main()
