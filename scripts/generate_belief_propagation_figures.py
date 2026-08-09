#!/usr/bin/env python3
"""Generate deterministic data and figures for the belief-propagation series."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "assets/data/belief-propagation/experiments.json"
OUT_DIR = ROOT / "assets/img/belief-propagation"

SURFACE = "#fcfcfb"
INK = "#172033"
SECONDARY = "#525b6b"
MUTED = "#7a8290"
GRID = "#e0e4ea"
BASELINE = "#c5cad3"
BLUE = "#2a78d6"
ORANGE = "#e56a32"
GREEN = "#168a65"
PURPLE = "#7a5cc7"
RED = "#c94a4a"

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
        "svg.hashsalt": "belief-propagation-figures-v1",
    }
)


def bisect_root(function: Callable[[float], float], left: float, right: float) -> float:
    """Find a bracketed scalar root without adding a SciPy dependency."""
    f_left = function(left)
    f_right = function(right)
    if f_left == 0:
        return left
    if f_right == 0:
        return right
    if f_left * f_right > 0:
        raise ValueError("root is not bracketed")
    for _ in range(100):
        middle = 0.5 * (left + right)
        f_middle = function(middle)
        if f_left * f_middle <= 0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    return 0.5 * (left + right)


def rounded(values: np.ndarray | list[float], digits: int = 10) -> list[float]:
    return [round(float(value), digits) for value in values]


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Date": None, "Creator": "generate_belief_propagation_figures.py"},
    )
    fig.savefig(
        OUT_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
        metadata={"Software": "generate_belief_propagation_figures.py"},
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hard-core model on the infinite regular tree
# ---------------------------------------------------------------------------
def hard_core_map(p: float, fugacity: float, degree: int) -> float:
    weight = fugacity * (1.0 - p) ** (degree - 1)
    return weight / (1.0 + weight)


def hard_core_fixed_point(fugacity: float, degree: int) -> float:
    return bisect_root(
        lambda p: hard_core_map(p, fugacity, degree) - p,
        0.0,
        1.0 - 1e-14,
    )


def hard_core_occupation(p: float, fugacity: float, degree: int) -> float:
    weight = fugacity * (1.0 - p) ** degree
    return weight / (1.0 + weight)


def hard_core_two_cycle(fugacity: float, degree: int) -> tuple[float, float]:
    p_a, p_b = 0.9, 0.1
    for _ in range(20_000):
        new_a = hard_core_map(p_b, fugacity, degree)
        new_b = hard_core_map(p_a, fugacity, degree)
        if abs(new_a - p_a) + abs(new_b - p_b) < 1e-13:
            return new_a, new_b
        p_a, p_b = new_a, new_b
    return p_a, p_b


def hard_core_data() -> dict:
    degree = 3
    threshold = (degree - 1) ** (degree - 1) / (degree - 2) ** degree
    fugacities = np.linspace(0.05, 10.0, 260)
    cavity = np.array([hard_core_fixed_point(lam, degree) for lam in fugacities])
    occupation = np.array(
        [hard_core_occupation(p, lam, degree) for p, lam in zip(cavity, fugacities)]
    )
    cycle = np.array([hard_core_two_cycle(lam, degree) for lam in fugacities])
    order = np.abs(cycle[:, 0] - cycle[:, 1])
    stability = (degree - 1) * cavity

    critical_p = hard_core_fixed_point(threshold, degree)
    assert abs((degree - 1) * critical_p - 1.0) < 1e-11

    return {
        "model": "hard-core model on the infinite 3-regular tree",
        "degree": degree,
        "fugacity_threshold": threshold,
        "threshold_meaning": (
            "uniqueness and local stability threshold of the translation-invariant "
            "tree recursion; not a universal finite-graph algorithmic threshold"
        ),
        "columns": {
            "fugacity": rounded(fugacities),
            "symmetric_cavity_occupation": rounded(cavity),
            "symmetric_node_occupation": rounded(occupation),
            "two_sublattice_order": rounded(order),
            "absolute_fixed_point_derivative": rounded(stability),
        },
    }


# ---------------------------------------------------------------------------
# Exact enumeration versus pairwise BP
# ---------------------------------------------------------------------------
def hard_core_exact_log_partition(n: int, edges: list[tuple[int, int]], fugacity: float) -> float:
    partition = 0.0
    for state in itertools.product((0, 1), repeat=n):
        if any(state[i] and state[j] for i, j in edges):
            continue
        partition += fugacity ** sum(state)
    return float(np.log(partition))


def pairwise_hard_core_bp(
    n: int,
    edges: list[tuple[int, int]],
    fugacity: float,
    damping: float = 0.35,
) -> tuple[float, int, float]:
    adjacency = {i: [] for i in range(n)}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    messages = {(i, j): np.array([0.5, 0.5]) for i, j in edges for i, j in ((i, j), (j, i))}
    factor = np.array([[1.0, 1.0], [1.0, 0.0]])
    node_factor = np.array([1.0, fugacity])
    residual = np.inf

    for iteration in range(1, 10_001):
        updated = {}
        for i, j in messages:
            product = node_factor.copy()
            for k in adjacency[i]:
                if k != j:
                    product *= factor.T @ messages[(k, i)]
            product /= product.sum()
            mixed = (1.0 - damping) * product + damping * messages[(i, j)]
            updated[(i, j)] = mixed / mixed.sum()
        residual = max(float(np.max(np.abs(updated[key] - messages[key]))) for key in messages)
        messages = updated
        if residual < 1e-13:
            break

    log_partition = 0.0
    for i in range(n):
        product = node_factor.copy()
        for k in adjacency[i]:
            product *= factor.T @ messages[(k, i)]
        log_partition += float(np.log(product.sum()))
    for i, j in edges:
        edge_normalizer = messages[(i, j)] @ factor @ messages[(j, i)]
        log_partition -= float(np.log(edge_normalizer))
    return log_partition, iteration, residual


def exact_vs_bethe_data() -> dict:
    graphs = {
        "four_node_path": {"n": 4, "edges": [(0, 1), (1, 2), (2, 3)], "topology": "tree"},
        "triangle": {"n": 3, "edges": [(0, 1), (1, 2), (0, 2)], "topology": "one loop"},
    }
    fugacities = np.geomspace(0.05, 20.0, 120)
    output = {
        "model": "finite hard-core model",
        "fugacity": rounded(fugacities),
        "graphs": {},
    }
    for name, graph in graphs.items():
        exact = []
        bethe = []
        iterations = []
        residuals = []
        for fugacity in fugacities:
            exact.append(hard_core_exact_log_partition(graph["n"], graph["edges"], fugacity))
            estimate, iteration, residual = pairwise_hard_core_bp(
                graph["n"], graph["edges"], fugacity
            )
            bethe.append(estimate)
            iterations.append(iteration)
            residuals.append(residual)
        errors = np.asarray(bethe) - np.asarray(exact)
        if graph["topology"] == "tree":
            assert np.max(np.abs(errors)) < 2e-10
        assert max(residuals) < 1e-12
        output["graphs"][name] = {
            "nodes": graph["n"],
            "edges": [list(edge) for edge in graph["edges"]],
            "topology": graph["topology"],
            "exact_log_partition": rounded(exact),
            "bethe_log_partition": rounded(bethe),
            "bethe_minus_exact": rounded(errors),
            "bp_iterations": iterations,
        }
    return output


# ---------------------------------------------------------------------------
# Bethe objective along a uniform-message family
# ---------------------------------------------------------------------------
def phi_uniform_field(a: float, degree: int, coupling: float, field: float) -> float:
    aligned, opposed = np.exp(coupling), np.exp(-coupling)
    incoming_up = aligned * a + opposed * (1.0 - a)
    incoming_down = opposed * a + aligned * (1.0 - a)
    node_normalizer = (
        np.exp(field) * incoming_up**degree
        + np.exp(-field) * incoming_down**degree
    )
    edge_normalizer = (
        aligned * (a * a + (1.0 - a) ** 2)
        + opposed * 2.0 * a * (1.0 - a)
    )
    return float(np.log(node_normalizer) - 0.5 * degree * np.log(edge_normalizer))


def magnetization_uniform_field(a: float, degree: int, coupling: float, field: float) -> float:
    aligned, opposed = np.exp(coupling), np.exp(-coupling)
    incoming_up = aligned * a + opposed * (1.0 - a)
    incoming_down = opposed * a + aligned * (1.0 - a)
    up = np.exp(field) * incoming_up**degree
    down = np.exp(-field) * incoming_down**degree
    return float((up - down) / (up + down))


def cavity_update_field(a: float, degree: int, coupling: float, field: float) -> float:
    aligned, opposed = np.exp(coupling), np.exp(-coupling)
    incoming_up = aligned * a + opposed * (1.0 - a)
    incoming_down = opposed * a + aligned * (1.0 - a)
    up = np.exp(field) * incoming_up ** (degree - 1)
    down = np.exp(-field) * incoming_down ** (degree - 1)
    return float(up / (up + down))


def scan_roots(function: Callable[[float], float]) -> list[float]:
    grid = np.linspace(1e-4, 1.0 - 1e-4, 6000)
    values = [function(value) for value in grid]
    roots = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0 or f_left * f_right < 0:
            root = bisect_root(function, float(left), float(right))
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)
    return roots


def bethe_landscape_data() -> dict:
    degree = 3
    coupling = 0.8
    field = 0.06
    cavity_grid = np.linspace(1e-4, 1.0 - 1e-4, 700)
    magnetization = np.array(
        [magnetization_uniform_field(a, degree, coupling, field) for a in cavity_grid]
    )
    objective = np.array(
        [-phi_uniform_field(a, degree, coupling, field) for a in cavity_grid]
    )
    order = np.argsort(magnetization)

    fixed_cavity = scan_roots(
        lambda a: cavity_update_field(a, degree, coupling, field) - a
    )
    epsilon = 1e-6
    fixed_points = []
    for a in fixed_cavity:
        slope = (
            cavity_update_field(a + epsilon, degree, coupling, field)
            - cavity_update_field(a - epsilon, degree, coupling, field)
        ) / (2.0 * epsilon)
        fixed_points.append(
            {
                "cavity_probability": round(a, 10),
                "magnetization": round(magnetization_uniform_field(a, degree, coupling, field), 10),
                "restricted_bethe_objective": round(-phi_uniform_field(a, degree, coupling, field), 10),
                "absolute_update_derivative": round(abs(slope), 10),
                "locally_stable": bool(abs(slope) < 1.0),
            }
        )
    assert sum(point["locally_stable"] for point in fixed_points) >= 2

    def trajectory(initial: float, steps: int = 45) -> list[float]:
        a = initial
        values = [magnetization_uniform_field(a, degree, coupling, field)]
        for _ in range(steps):
            a = cavity_update_field(a, degree, coupling, field)
            values.append(magnetization_uniform_field(a, degree, coupling, field))
        return values

    aligned = trajectory(0.95)
    anti_aligned = trajectory(0.05)
    return {
        "model": "ferromagnetic Ising model in a uniform-message reduction",
        "degree": degree,
        "beta_times_coupling": coupling,
        "beta_times_field": field,
        "scope": (
            "one-dimensional restriction of the Bethe normalizer expression to "
            "uniform cavity messages, not the full Bethe variational domain"
        ),
        "curve": {
            "magnetization": rounded(magnetization[order]),
            "restricted_bethe_objective": rounded(objective[order]),
        },
        "fixed_points": fixed_points,
        "flow": {
            "iteration": list(range(len(aligned))),
            "aligned_initialization": rounded(aligned),
            "anti_aligned_initialization": rounded(anti_aligned),
        },
    }


# ---------------------------------------------------------------------------
# Parallel-update convergence on a frustrated spin glass
# ---------------------------------------------------------------------------
def random_regular_edges(n: int, degree: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Configuration-model sample, retried until it is a simple graph."""
    while True:
        stubs = np.repeat(np.arange(n), degree)
        rng.shuffle(stubs)
        edges: set[tuple[int, int]] = set()
        ok = True
        for a in range(0, len(stubs), 2):
            u, v = int(stubs[a]), int(stubs[a + 1])
            if u == v or (min(u, v), max(u, v)) in edges:
                ok = False
                break
            edges.add((min(u, v), max(u, v)))
        if ok and len(edges) == n * degree // 2:
            return sorted(edges)


def spin_glass_residuals(
    damping: float,
    n: int = 60,
    degree: int = 3,
    coupling: float = 2.0,
    seed: int = 1,
    iterations: int = 3000,
) -> list[float]:
    """Per-sweep max message change for parallel BP on a +/-J Ising spin glass."""
    rng = np.random.default_rng(seed)
    edges = random_regular_edges(n, degree, rng)
    signs = {edge: (1.0 if rng.random() < 0.5 else -1.0) for edge in edges}
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    factors = {}
    for edge in edges:
        b = coupling * signs[edge]
        factors[edge] = np.array([[np.exp(b), np.exp(-b)], [np.exp(-b), np.exp(b)]])

    def oriented(i: int, j: int) -> np.ndarray:
        edge = (min(i, j), max(i, j))
        matrix = factors[edge]
        return matrix if (i, j) == edge else matrix.T

    messages = {}
    for i, j in edges:
        for a, b in ((i, j), (j, i)):
            v = rng.random(2)
            messages[(a, b)] = v / v.sum()

    residuals = []
    for _ in range(iterations):
        updated = {}
        for i, j in messages:
            product = np.ones(2)
            for k in adjacency[i]:
                if k != j:
                    product = product * (oriented(k, i).T @ messages[(k, i)])
            total = product.sum()
            product = product / total if total > 0 else np.ones(2) / 2.0
            mixed = (1.0 - damping) * product + damping * messages[(i, j)]
            updated[(i, j)] = mixed / mixed.sum()
        residuals.append(
            max(float(np.max(np.abs(updated[key] - messages[key]))) for key in messages)
        )
        messages = updated
    return residuals


def spin_glass_data() -> dict:
    dampings = [0.0, 0.5, 0.8, 0.9, 0.95]
    runs = {}
    for damping in dampings:
        residuals = spin_glass_residuals(damping)
        runs[f"{damping:.2f}"] = {
            "damping": damping,
            "final_residual": round(residuals[-1], 10),
            "converged_below_1e-10": bool(residuals[-1] < 1e-10),
            "residual_trace": rounded(residuals[::10]),
        }
    assert not any(run["converged_below_1e-10"] for run in runs.values())
    return {
        "model": "random 3-regular +/-J Ising spin glass, 60 spins, beta*|J| = 2",
        "seed": 1,
        "iterations": 3000,
        "schedule": "synchronous (parallel) updates with damping",
        "note": (
            "damping lowers the residual monotonically but does not reach the "
            "tolerance; the run is a non-convergence demonstration, not a claim "
            "about all spin glasses"
        ),
        "runs": runs,
    }


def generate_data() -> dict:
    return {
        "schema_version": 1,
        "generated_by": "scripts/generate_belief_propagation_figures.py",
        "provenance": {
            "description": "Deterministic calculations used by the belief-propagation article figures.",
            "random_seed": "fixed per-experiment; see each section",
            "floating_point": "IEEE-754 double precision via NumPy",
        },
        "hard_core_tree_transition": hard_core_data(),
        "exact_vs_bethe": exact_vs_bethe_data(),
        "bethe_landscape": bethe_landscape_data(),
        "spin_glass_convergence": spin_glass_data(),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def hard_core_transition_figure(data: dict) -> None:
    columns = data["columns"]
    fugacity = np.asarray(columns["fugacity"])
    occupation = np.asarray(columns["symmetric_node_occupation"])
    order = np.asarray(columns["two_sublattice_order"])
    stability = np.asarray(columns["absolute_fixed_point_derivative"])
    threshold = data["fugacity_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    axes[0].plot(fugacity, occupation, color=BLUE, linewidth=2.4, label="node occupation")
    axes[0].plot(fugacity, order, color=ORANGE, linewidth=2.4, label="two-sublattice order")
    axes[0].axvline(threshold, color=SECONDARY, linewidth=1.2, linestyle="--")
    axes[0].annotate(
        r"$\lambda_c=4$",
        (threshold, 0.04),
        xytext=(7, 0),
        textcoords="offset points",
        color=SECONDARY,
        fontsize=10,
    )
    axes[0].set(xlabel=r"fugacity $\lambda$", ylabel="probability / order parameter", xlim=(0, 10), ylim=(0, 1))
    axes[0].legend(frameon=False, loc="upper left")
    style_axis(axes[0])

    axes[1].plot(fugacity, stability, color=PURPLE, linewidth=2.4, label=r"$|f'(p^\star)|$")
    axes[1].axhline(1.0, color=SECONDARY, linewidth=1.2, linestyle="--")
    axes[1].axvline(threshold, color=SECONDARY, linewidth=1.2, linestyle="--")
    axes[1].fill_between(fugacity, 1.0, stability, where=stability >= 1.0, color=PURPLE, alpha=0.12)
    axes[1].set(xlabel=r"fugacity $\lambda$", ylabel="linearized amplification", xlim=(0, 10), ylim=(0, 1.8))
    axes[1].text(7.7, 1.48, "unstable\nsymmetric recursion", ha="center", color=PURPLE, fontsize=10)
    axes[1].legend(frameon=False, loc="upper left")
    style_axis(axes[1])

    fig.suptitle("One scalar recursion reveals the hard-core tree transition", x=0.07, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.07, 0.90, "Infinite 3-regular tree · the dashed line is the tree uniqueness/local-stability threshold", color=SECONDARY, fontsize=10.5)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.17, wspace=0.27)
    save_figure(fig, "hard-core-tree-transition")


def exact_vs_bethe_figure(data: dict) -> None:
    fugacity = np.asarray(data["fugacity"])
    tree_error = np.asarray(data["graphs"]["four_node_path"]["bethe_minus_exact"])
    loop_error = np.asarray(data["graphs"]["triangle"]["bethe_minus_exact"])

    fig, ax = plt.subplots(figsize=(9.7, 4.9))
    ax.axhline(0.0, color=BASELINE, linewidth=1.2)
    ax.plot(fugacity, tree_error, color=GREEN, linewidth=2.4, label="four-node path (tree)")
    ax.plot(fugacity, loop_error, color=RED, linewidth=2.4, label="triangle (one loop)")
    ax.set_xscale("log")
    ax.set(xlabel=r"hard-core fugacity $\lambda$ (log scale)", ylabel=r"$\log Z_{\mathrm{Bethe}}-\log Z_{\mathrm{exact}}$")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)
    ax.annotate("tree: exact to numerical precision", (8.0, 0.0), xytext=(0, 15), textcoords="offset points", ha="center", color=GREEN, fontsize=10)
    ax.annotate("the missing loop correction grows", (6.3, loop_error[np.argmin(np.abs(fugacity - 6.3))]), xytext=(-8, -31), textcoords="offset points", ha="right", color=RED, fontsize=10, arrowprops={"arrowstyle": "-", "color": RED})

    fig.suptitle("The same BP equations are exact on a tree and approximate on a loop", x=0.10, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.10, 0.89, "Every point compares a converged BP fixed point with complete enumeration of the finite model", color=SECONDARY, fontsize=10.5)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.77, bottom=0.18)
    save_figure(fig, "exact-vs-bethe-loop-error")


def bethe_landscape_figure(data: dict) -> None:
    curve = data["curve"]
    flow = data["flow"]
    magnetization = np.asarray(curve["magnetization"])
    objective = np.asarray(curve["restricted_bethe_objective"])
    fixed_points = data["fixed_points"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    axes[0].plot(magnetization, objective, color=BLUE, linewidth=2.4)
    for point in fixed_points:
        color = GREEN if point["locally_stable"] else RED
        marker = "o" if point["locally_stable"] else "X"
        axes[0].scatter(point["magnetization"], point["restricted_bethe_objective"], s=85, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    axes[0].set(xlabel="magnetization", ylabel="restricted Bethe objective")
    style_axis(axes[0])

    iterations = np.asarray(flow["iteration"])
    axes[1].plot(iterations, flow["aligned_initialization"], color=GREEN, linewidth=2.4, label="field-aligned start")
    axes[1].plot(iterations, flow["anti_aligned_initialization"], color=ORANGE, linewidth=2.4, label="anti-aligned start")
    axes[1].set(xlabel="parallel BP iteration", ylabel="magnetization", xlim=(0, 45), ylim=(-1, 1))
    axes[1].legend(frameon=False, loc="center right")
    style_axis(axes[1])

    fixed_legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=GREEN, markeredgecolor=SURFACE, markersize=8, label="stable fixed point"),
        Line2D([0], [0], marker="X", linestyle="none", markerfacecolor=RED, markeredgecolor=SURFACE, markersize=8, label="unstable fixed point"),
    ]
    axes[0].legend(handles=fixed_legend, frameon=False, loc="upper center")
    fig.suptitle("Convergence can depend on initialization", x=0.07, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.07, 0.90, r"3-regular ferromagnetic Ising reduction · $\beta J=0.8$, $\beta h=0.06$", color=SECONDARY, fontsize=10.5)
    fig.text(0.07, 0.025, "Left: the Bethe normalizer expression restricted to uniform cavity messages; the full Bethe domain is higher-dimensional.", color=SECONDARY, fontsize=9.2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.19, wspace=0.28)
    save_figure(fig, "bethe-landscape-and-flow")


def spin_glass_figure(data: dict) -> None:
    runs = data["runs"]
    fig, ax = plt.subplots(figsize=(9.7, 4.9))
    palette = [RED, ORANGE, PURPLE, BLUE, GREEN]
    for (key, run), color in zip(sorted(runs.items()), palette):
        trace = np.asarray(run["residual_trace"])
        sweeps = np.arange(len(trace)) * 10
        ax.plot(sweeps, trace, color=color, linewidth=2.0)
        ax.annotate(
            f"damping {run['damping']:g}  →  {run['final_residual']:.1e}",
            (sweeps[-1], trace[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=9.5,
            fontweight="semibold",
        )
    ax.set_yscale("log")
    ax.set(xlabel="parallel BP sweep", ylabel="max message change (log scale)", xlim=(0, 3050))
    style_axis(ax)

    fig.suptitle("Damping slows the oscillation without stopping it", x=0.10, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.10, 0.89, r"Random 3-regular $\pm J$ Ising spin glass · 60 spins · $\beta|J|=2$ · synchronous updates", color=SECONDARY, fontsize=10.5)
    fig.text(0.10, 0.03, "No setting shown reaches the $10^{-10}$ tolerance in 3000 sweeps. This is one instance, not a claim about all spin glasses.", color=SECONDARY, fontsize=9.2)
    fig.subplots_adjust(left=0.11, right=0.72, top=0.77, bottom=0.19)
    save_figure(fig, "spin-glass-nonconvergence")


def main() -> None:
    data = generate_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    hard_core_transition_figure(data["hard_core_tree_transition"])
    exact_vs_bethe_figure(data["exact_vs_bethe"])
    bethe_landscape_figure(data["bethe_landscape"])
    spin_glass_figure(data["spin_glass_convergence"])


if __name__ == "__main__":
    main()
