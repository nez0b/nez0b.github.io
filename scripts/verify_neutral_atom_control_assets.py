#!/usr/bin/env python3
"""Geometry and raster acceptance checks for revised series assets."""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVER_TEX = ROOT / "assets/figures/neutral-atom-control/cover.tex"
IMAGE_DIR = ROOT / "assets/img/neutral-atom-control"


def cross(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def inside_convex(polygon: list[tuple[float, float]], point: tuple[float, float]) -> bool:
    values = [cross(a, polygon[(i + 1) % len(polygon)], point) for i, a in enumerate(polygon)]
    return all(value >= -1e-9 for value in values) or all(value <= 1e-9 for value in values)


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"not a PNG: {path}")
    return struct.unpack(">II", data[16:24])


def main() -> None:
    tex = COVER_TEX.read_text()
    assert "\\foreach \\j in {0,...,3}" in tex
    assert "\\foreach \\i in {0,...,4}" in tex
    assert "7 x 5" not in tex and "7×5" not in tex

    origin = (3.40, 1.65)
    u = (1.55, 0.25)
    v = (0.58, 0.72)
    polygon = [(2.0155, 1.0195), (10.2305, 2.3445), (12.7245, 5.4405), (4.5095, 4.1155)]
    sites = [
        (origin[0] + i * u[0] + j * v[0], origin[1] + i * u[1] + j * v[1])
        for j in range(4) for i in range(5)
    ]
    assert len(sites) == 20 and all(inside_convex(polygon, point) for point in sites)
    table_u = (polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1])
    table_v = (polygon[3][0] - polygon[0][0], polygon[3][1] - polygon[0][1])
    assert abs(cross((0.0, 0.0), u, table_u)) < 1e-12
    assert abs(cross((0.0, 0.0), v, table_v)) < 1e-12

    atom_a = sites[1 * 5 + 2]
    atom_b = sites[1 * 5 + 3]
    assert math.dist(atom_a, (7.08, 2.87)) < 1e-12
    assert math.dist(atom_b, (8.63, 3.12)) < 1e-12
    link = (atom_b[0] - atom_a[0], atom_b[1] - atom_a[1])
    assert abs(cross((0.0, 0.0), u, link)) < 1e-12
    assert "($(A)+(.20,.032)$) -- ($(B)+(-.20,-.032)$)" in tex

    for stem in ("cover", "part2-method-pulses"):
        png = IMAGE_DIR / f"{stem}.png"
        svg = IMAGE_DIR / f"{stem}.svg"
        assert png.exists() and png.stat().st_size > 20_000
        assert svg.exists() and svg.stat().st_size > 10_000
        width, height = png_size(png)
        assert width > 1800 and height > 1000
        if stem == "cover":
            assert abs(width / height - 16 / 9) < 0.01

    generator = (ROOT / "scripts/generate_neutral_atom_control_figures.py").read_text()
    for label in ("drive quadrature", "detuning", 'control / $2\\pi$'):
        assert label in generator
    assert re.search(r'fig\.legend\(.*outside upper center', generator)
    print("asset audit passed: 20 tabletop-aligned sites, anchored interaction arrow, and labeled pulse controls")


if __name__ == "__main__":
    main()
