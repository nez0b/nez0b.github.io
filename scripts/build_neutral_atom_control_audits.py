#!/usr/bin/env python3
"""Build the sanitized numerical audits used by the neutral-atom series.

This script only propagates saved pulse arrays.  It never optimizes a pulse and
never contacts a device.  The three JSON outputs are consumed by the figure
generator so the published plots do not depend on the notebook environments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quadrature_variation(omega_x: np.ndarray, omega_y: np.ndarray, scale: float) -> float:
    """L1 path length in the quadrature plane, normalized by Omega_max."""
    return float((np.abs(np.diff(omega_x)).sum() + np.abs(np.diff(omega_y)).sum()) / scale)


def load_figshare_csv(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["|Omega|"]]
    return {
        "t_omega_max": [float(row["t"]) for row in rows],
        "amplitude_over_omega_max": [float(row["|Omega|"]) for row in rows],
        "phase_rad": [float(row["arg(Omega)"]) for row in rows],
    }


def model(g1: dict):
    p1r = np.zeros((3, 3), complex)
    p1r[1, 2] = 1.0
    nr = np.diag([0.0, 0.0, 1.0]).astype(complex)
    i3 = np.eye(3, dtype=complex)
    hx = 0.5 * (
        np.kron(p1r + p1r.conj().T, i3) + np.kron(i3, p1r + p1r.conj().T)
    )
    hy = 0.5 * (
        np.kron(1j * p1r - 1j * p1r.conj().T, i3)
        + np.kron(i3, 1j * p1r - 1j * p1r.conj().T)
    )
    hdelta = -(np.kron(nr, i3) + np.kron(i3, nr))
    h0 = g1["constants"]["V_canonical"] * np.kron(nr, nr)
    ntot = np.kron(nr, i3) + np.kron(i3, nr)
    return p1r, nr, hx, hy, hdelta, h0, ntot


def liouvillian(h: np.ndarray, c_ops: list[np.ndarray]) -> np.ndarray:
    ident = np.eye(h.shape[0], dtype=complex)
    result = -1j * (np.kron(ident, h) - np.kron(h.T, ident))
    for c in c_ops:
        cdc = c.conj().T @ c
        result += (
            np.kron(c.conj(), c)
            - 0.5 * np.kron(ident, cdc)
            - 0.5 * np.kron(cdc.T, ident)
        )
    return result


def pulse_map(pulse: dict, components: tuple[np.ndarray, ...], c_ops: list[np.ndarray]) -> np.ndarray:
    hx, hy, hdelta, h0 = components
    result = np.eye(81, dtype=complex)
    for x, y, delta in zip(pulse["omega_x"], pulse["omega_y"], pulse["delta"]):
        h = h0 + x * hx + y * hy + delta * hdelta
        result = expm(liouvillian(h, c_ops) * pulse["dt_us"]) @ result
    return result


def apply_map(superop: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return (superop @ rho.reshape(-1, order="F")).reshape(9, 9, order="F")


def process_survival(superop: np.ndarray, theta: float) -> tuple[float, float]:
    comp = [0, 1, 3, 4]
    target = np.diag([1.0, np.exp(1j * theta), np.exp(1j * theta), -np.exp(2j * theta)])
    process = 0.0j
    survival = 0.0
    for i in range(4):
        for j in range(4):
            embedded = np.zeros((9, 9), complex)
            embedded[comp[i], comp[j]] = 1.0
            output = apply_map(superop, embedded)[np.ix_(comp, comp)]
            basis = np.zeros((4, 4), complex)
            basis[i, j] = 1.0
            desired = target @ basis @ target.conj().T
            process += np.trace(desired.conj().T @ output)
            if i == j:
                survival += np.trace(output).real
    return float(process.real / 16.0), float(survival / 4.0)


def best_process(superop: np.ndarray) -> tuple[float, float, float]:
    grid = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    values = np.array([process_survival(superop, theta)[0] for theta in grid])
    theta0 = float(grid[int(np.argmax(values))])
    fit = minimize_scalar(
        lambda theta: -process_survival(superop, theta)[0],
        bracket=(theta0 - 0.04, theta0, theta0 + 0.04),
        options={"xtol": 1e-13},
    )
    process, survival = process_survival(superop, float(fit.x))
    return process, survival, float(fit.x % (2 * np.pi))


def coherent_trace_fidelity(pulse: dict, components: tuple[np.ndarray, ...], epsilon: float) -> float:
    hx, hy, hdelta, h0 = components
    unitary = np.eye(9, dtype=complex)
    for x, y, delta in zip(pulse["omega_x"], pulse["omega_y"], pulse["delta"]):
        h = h0 + (1 + epsilon) * (x * hx + y * hy) + delta * hdelta
        unitary = expm(-1j * h * pulse["dt_us"]) @ unitary
    comp = [0, 1, 3, 4]
    block = unitary[np.ix_(comp, comp)]
    grid = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    def fidelity(theta: float) -> float:
        target = np.diag([1.0, np.exp(1j * theta), np.exp(1j * theta), -np.exp(2j * theta)])
        return float(abs(np.trace(target.conj().T @ block)) ** 2 / 16.0)

    theta0 = float(grid[int(np.argmax([fidelity(t) for t in grid]))])
    fit = minimize_scalar(lambda theta: -fidelity(theta), bracket=(theta0 - 0.03, theta0, theta0 + 0.03))
    return float(-fit.fun)


def exposure(pulse: dict, components: tuple[np.ndarray, ...], ntot: np.ndarray) -> float:
    hx, hy, hdelta, h0 = components
    total = 0.0
    for basis_index in (0, 1, 3, 4):
        psi = np.eye(9, dtype=complex)[:, basis_index]
        for x, y, delta in zip(pulse["omega_x"], pulse["omega_y"], pulse["delta"]):
            h = h0 + x * hx + y * hy + delta * hdelta
            psi = expm(-1j * h * pulse["dt_us"]) @ psi
            total += float(np.real(psi.conj() @ ntot @ psi)) * pulse["dt_us"]
    return total / 4.0


def pulse_families(g2: dict, g3: dict, g5: dict) -> dict[str, dict]:
    families = {
        "Levine–Pichler": dict(g2["lp_cz"]),
        "CRAB": dict(g2["crab_cz"]),
        "GRAPE": dict(g2["grape_cz"]),
        "Krotov": dict(g2["krotov_cz"]),
        "open-system GRAPE": {**g5["open_grape"], "delta": [0.0] * g5["open_grape"]["N"]},
        "robust GRAPE": {**g5["robust_grape"], "delta": [0.0] * g5["robust_grape"]["N"]},
    }
    for floor, record in g3["mintime"].items():
        ux = np.asarray(record["omega_x"])
        uy = np.asarray(record["omega_y"])
        families[f"collocation {floor}"] = {
            "N": len(ux) - 1,
            "T_us": record["T_us"],
            "dt_us": record["dt_us"],
            "omega_x": (0.5 * (ux[:-1] + ux[1:])).tolist(),
            "omega_y": (0.5 * (uy[:-1] + uy[1:])).tolist(),
            "delta": [0.0] * (len(ux) - 1),
        }
    return families


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "assets/data/neutral-atom-control")
    parser.add_argument(
        "--jp-dir",
        type=Path,
        default=Path("/Users/nez0b/Code/Quantum/harmoniq-Pasqal/jandura-pupillo-reproduce"),
    )
    args = parser.parse_args()
    g1 = read_json(args.data_dir / "g1_baselines.json")
    g2 = read_json(args.data_dir / "g2_pulses.json")
    g3 = read_json(args.data_dir / "g3_collocation.json")
    g5 = read_json(args.data_dir / "g5_noise.json")

    jp_files = {
        "time_optimal_cz": args.jp_dir / "figshare/extracted/01_cz.csv",
        "finite_blockade_robust_cz": args.jp_dir / "figshare/extracted/12_cz_robust_B.csv",
    }
    all_scores = read_json(args.jp_dir / "results/figshare_scores.json")
    selected_scores = [
        row for row in all_scores["results"] if row["file"] in {"01_cz.csv", "12_cz_robust_B.csv"}
    ]
    jp = {
        "meta": {
            "source": "Jandura–Pupillo Figshare dataset 19658427",
            "units": "dimensionless time t*Omega_max, amplitude/Omega_max, phase in rad",
            "transformation": "blank terminal delimiter row removed; numeric columns copied",
            "source_sha256": {name: sha256(path) for name, path in jp_files.items()},
        },
        "pulses": {name: load_figshare_csv(path) for name, path in jp_files.items()},
        "independent_scores": {
            "fidelity": all_scores["fidelity"],
            "results": selected_scores,
            "transformation": "only the two vendored pulse records retained",
        },
    }
    write_json(args.data_dir / "jp_figshare_cz.json", jp)

    omega_max = g1["constants"]["omega_max"]
    audits = []
    for name, key in (
        ("GRAPE", "grape_cz"),
        ("Krotov", "krotov_cz"),
        ("CRAB", "crab_cz"),
        ("analytic Levine–Pichler", "lp_cz"),
    ):
        p = g2[key]
        audits.append(
            {
                "name": name,
                "definition": "sum_k(|ΔOmega_x|+|ΔOmega_y|)/Omega_max",
                "normalized_quadrature_variation": quadrature_variation(
                    np.asarray(p["omega_x"]), np.asarray(p["omega_y"]), omega_max
                ),
            }
        )
    published = jp["pulses"]["time_optimal_cz"]
    amp = np.asarray(published["amplitude_over_omega_max"])
    phase = np.asarray(published["phase_rad"])
    audits.append(
        {
            "name": "Jandura–Pupillo 01_cz",
            "definition": "sum_k(|ΔOmega_x|+|ΔOmega_y|)/Omega_max",
            "normalized_quadrature_variation": quadrature_variation(
                amp * np.cos(phase), amp * np.sin(phase), 1.0
            ),
        }
    )
    write_json(
        args.data_dir / "pulse_smoothness_audit.json",
        {
            "meta": {
                "source": "saved pulse-search arrays and sanitized Jandura–Pupillo 01_cz",
                "units": "dimensionless",
                "transformation": "no smoothing or resampling; endpoint differences only",
            },
            "pulses": audits,
        },
    )

    p1r, nr, hx, hy, hdelta, h0, ntot = model(g1)
    i3 = np.eye(3, dtype=complex)
    c_ops = [
        np.sqrt(0.01) * np.kron(p1r, i3),
        np.sqrt(0.01) * np.kron(i3, p1r),
        np.sqrt(0.05) * np.kron(nr, i3),
        np.sqrt(0.05) * np.kron(i3, nr),
    ]
    components = (hx, hy, hdelta, h0)
    legacy_by_name = {
        row["name"]
        .replace("Levine-Pichler", "Levine–Pichler")
        .replace("G3 min-time", "collocation"): row
        for row in g5["family_scores"]
    }
    legacy_by_name["open-system GRAPE"] = g5["open_grape"]
    scored = []
    families = pulse_families(g2, g3, g5)
    for name, pulse in families.items():
        superop = pulse_map(pulse, components, c_ops)
        process, survival, theta = best_process(superop)
        corrected = (4 * process + survival) / 5
        legacy = legacy_by_name.get(name, {})
        legacy_fidelity = 1 - legacy["noisy_infid"] if "noisy_infid" in legacy else None
        scored.append(
            {
                "name": name,
                "duration_us": len(pulse["omega_x"]) * pulse["dt_us"],
                "target_theta_rad": theta,
                "process_fidelity": process,
                "survival": survival,
                "average_fidelity_unconditional": corrected,
                "coherent_trace_infidelity": 1 - coherent_trace_fidelity(pulse, components, 0.0),
                "legacy_average_fidelity": legacy_fidelity,
                "rydberg_exposure_us": exposure(pulse, components, ntot),
            }
        )

    eps = np.linspace(-0.03, 0.03, 61)
    responses = {}
    for name in ("GRAPE", "robust GRAPE"):
        pulse = families[name]
        responses[name] = {
            "epsilon": eps.tolist(),
            "trace_infidelity": [1 - coherent_trace_fidelity(pulse, components, float(e)) for e in eps],
        }
    write_json(
        args.data_dir / "g5_leakage_aware_scores.json",
        {
            "meta": {
                "source": "saved coherent, collocation, open-system, and robust-control arrays",
                "model": "two-qutrit Lindblad; relaxation 0.01/us and Rydberg dephasing 0.05/us per atom",
                "fidelity": "Favg_uncond=(4 Fpro+s)/5, s=Tr[P E(P)]/4",
                "transformation": "independent piecewise-constant propagation; no optimization",
            },
            "scores": scored,
            "amplitude_error_response": responses,
        },
    )

    for filename in ("jp_figshare_cz.json", "pulse_smoothness_audit.json", "g5_leakage_aware_scores.json"):
        print(f"{filename}: {sha256(args.data_dir / filename)}")


if __name__ == "__main__":
    main()
