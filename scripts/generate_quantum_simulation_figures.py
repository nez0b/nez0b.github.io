#!/usr/bin/env python3
"""Generate deterministic performance figures for the quantum-simulation series."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "assets/data/quantum-simulation/statevector-bitshift-benchmarks.json"
OUT_DIR = ROOT / "assets/img/quantum-simulation"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COLORS = {
    "oracle": "#2a78d6",
    "reshape": "#eb6834",
    "bitshift": "#1baf7a",
}
BACKEND_ORDER = ["oracle", "reshape", "bitshift"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED,
        "ytick.color": SECONDARY,
        "text.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "quantum-simulation-figures-v1",
    }
)


def load_data() -> dict:
    return json.loads(DATA_PATH.read_text())


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Date": None, "Creator": "generate_quantum_simulation_figures.py"},
    )
    fig.savefig(
        OUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Software": "generate_quantum_simulation_figures.py"},
    )
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", length=0)


def format_runtime(microseconds: float) -> str:
    if microseconds < 10:
        return f"{microseconds:.2f} µs"
    return f"{microseconds:.1f} µs"


def rust_backend_figure(data: dict) -> None:
    rows = data["rust_single_gate"]["measurements"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    y = np.arange(len(BACKEND_ORDER))[::-1]

    display_names = {
        row["backend"]: row["display_name"]
        for row in rows
    }

    for ax, qubits in zip(axes, (12, 16)):
        subset = {row["backend"]: row for row in rows if row["qubits"] == qubits}
        for ypos, backend in zip(y, BACKEND_ORDER):
            row = subset[backend]
            mean = row["mean_ns"] / 1000.0
            lower = row["ci95_lower_ns"] / 1000.0
            upper = row["ci95_upper_ns"] / 1000.0
            ax.errorbar(
                mean,
                ypos,
                xerr=[[mean - lower], [upper - mean]],
                fmt="o",
                markersize=9,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                color=COLORS[backend],
                ecolor=COLORS[backend],
                elinewidth=2,
                capsize=4,
                capthick=1.5,
                zorder=3,
            )
            ax.annotate(
                format_runtime(mean),
                (mean, ypos),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=10,
                fontweight="semibold",
            )

        ax.set_xscale("log")
        ax.set_title(f"{qubits} qubits  ·  {1 << qubits:,} amplitudes", loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time for one Hadamard gate (µs, log scale)")
        ax.set_yticks(y, [display_names[key] for key in BACKEND_ORDER])
        style_axis(ax)

    fig.suptitle("Direct pair indexing cuts allocation and traversal overhead", x=0.075, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.075,
        0.895,
        "Criterion arithmetic mean with 95% confidence intervals · Apple M3 Pro · one thread",
        color=SECONDARY,
        fontsize=10.5,
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8, markerfacecolor=COLORS[key], markeredgecolor=SURFACE, label=display_names[key])
        for key in BACKEND_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, -0.01))
    fig.subplots_adjust(left=0.19, right=0.96, top=0.78, bottom=0.22, wspace=0.26)
    save_figure(fig, "rust-backend-comparison")


def quest_dilution_figure(data: dict) -> None:
    levels = data["quest_bmi2_case_study"]["levels"]
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    y = np.arange(len(levels))[::-1]
    stage_colors = [COLORS["oracle"], COLORS["reshape"], COLORS["bitshift"]]

    ax.axvline(1.0, color=BASELINE, linewidth=1.2, zorder=0)
    for ypos, level, color in zip(y, levels, stage_colors):
        lo = level["min_speedup"]
        hi = level["max_speedup"]
        rep = level["representative_speedup"]
        ax.plot([lo, hi], [ypos, ypos], color=color, linewidth=7, solid_capstyle="round", zorder=2)
        ax.scatter([rep], [ypos], s=105, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(
            f"{lo:.2g}–{hi:.2g}×",
            (hi, ypos),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.9, 15.5)
    ax.set_xticks([1, 2, 3, 5, 10], ["1×", "2×", "3×", "5×", "10×"])
    ax.set_yticks(y, [level["display_name"] for level in levels])
    ax.set_xlabel("Speedup over the scalar implementation (log scale)")
    style_axis(ax)
    ax.text(1.0, y[-1] - 0.58, "no speedup", ha="center", va="top", fontsize=9, color=MUTED)

    fig.suptitle("A faster index instruction is diluted by statevector traffic", x=0.10, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.10,
        0.89,
        "QuEST PR #796 · Xeon Gold 6448H · ranges summarize distinct measurement levels",
        color=SECONDARY,
        fontsize=10.5,
    )
    fig.text(
        0.10,
        0.04,
        "Isolated: hoisted PDEP/PEXT helpers · Kernel: one-thread q=18 scatter/gather · Circuits: eight 12–16-qubit workloads",
        color=SECONDARY,
        fontsize=9.3,
    )
    fig.subplots_adjust(left=0.27, right=0.92, top=0.77, bottom=0.23)
    save_figure(fig, "quest-optimization-dilution")


def main() -> None:
    data = load_data()
    rust_backend_figure(data)
    quest_dilution_figure(data)


if __name__ == "__main__":
    main()
