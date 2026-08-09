#!/usr/bin/env python3
"""Numerical acceptance checks for the neutral-atom series artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets/data/neutral-atom-control"


def read(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def main() -> None:
    g1 = read("g1_baselines.json")
    g2 = read("g2_pulses.json")
    jp = read("jp_figshare_cz.json")
    rough = read("pulse_smoothness_audit.json")
    g5 = read("g5_leakage_aware_scores.json")
    c = g1["constants"]
    assert np.isclose(c["V_canonical"], c["C6_rad_per_us_um6"] / c["r_canonical_um"] ** 6)
    blockade_ratio = c["V_canonical"] / (np.sqrt(2) * c["omega_max"])
    assert np.isclose(blockade_ratio, 3.1176982407)

    p = np.zeros((3, 3), complex); p[1, 2] = 1
    nr = np.diag([0, 0, 1]).astype(complex); ident = np.eye(3)
    h_parts = [
        .5 * (np.kron(p + p.T, ident) + np.kron(ident, p + p.T)),
        .5 * (np.kron(1j*p - 1j*p.T, ident) + np.kron(ident, 1j*p - 1j*p.T)),
        -(np.kron(nr, ident) + np.kron(ident, nr)),
        c["V_canonical"] * np.kron(nr, nr),
    ]
    assert all(np.max(abs(h - h.conj().T)) < 1e-14 for h in h_parts)

    for theta in np.linspace(-np.pi, np.pi, 11):
        phases = np.array([0, theta, theta, np.pi + 2 * theta])
        entangling = phases[0] - phases[1] - phases[2] + phases[3]
        assert np.isclose(np.exp(1j * entangling), -1)

    score = {row["file"]: row for row in jp["independent_scores"]["results"]}
    assert np.isclose(score["01_cz.csv"]["T_Omega_max"], 7.612)
    assert score["01_cz.csv"]["infidelity_tr"] < 1e-9
    assert np.isclose(score["12_cz_robust_B.csv"]["B"], 10)
    assert score["12_cz_robust_B.csv"]["F_tr_coset"] > .9998

    rv = {row["name"]: row["normalized_quadrature_variation"] for row in rough["pulses"]}
    expected = {"GRAPE": 26.4117, "Krotov": 73.4866, "CRAB": 3.6428,
                "Jandura–Pupillo 01_cz": 4.45025}
    for name, value in expected.items():
        assert np.isclose(rv[name], value, rtol=2e-5)

    assert all(v >= 0 for key in ("infid_m2", "infid_m3") for v in g2["frontier"][key])
    for row in g5["scores"]:
        expected_favg = (4 * row["process_fidelity"] + row["survival"]) / 5
        assert np.isclose(row["average_fidelity_unconditional"], expected_favg, atol=1e-13)
        assert 0 <= row["survival"] <= 1 + 1e-12
    ordered = sorted(g5["scores"], key=lambda row: 1 - row["average_fidelity_unconditional"])
    assert [row["name"] for row in ordered[:2]] == ["CRAB", "open-system GRAPE"]

    print(
        "numerical audit passed: Hermitian Hamiltonian, "
        f"blockade ratio {blockade_ratio:.6f}, CZ coset, JP scores, roughness, and leakage fidelity"
    )


if __name__ == "__main__":
    main()
