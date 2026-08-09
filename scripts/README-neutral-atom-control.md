# Neutral-atom series figure build

The quantitative figures use only the sanitized JSON files under
<code>assets/data/neutral-atom-control/</code>; they do not rerun an optimizer or
contact a QPU.

    python -m pip install -r scripts/requirements-neutral-atom-control.txt
    python scripts/build_neutral_atom_control_audits.py
    python scripts/generate_neutral_atom_control_figures.py
    make -C assets/figures/neutral-atom-control
    python scripts/count_neutral_atom_control_words.py
    python scripts/lint_neutral_atom_control_series.py
    python scripts/verify_neutral_atom_control_data.py
    python scripts/verify_neutral_atom_control_assets.py
    python scripts/verify_neutral_atom_control_build.py --build-dir PATH_TO_JEKYLL_BUILD

The Python generator writes SVG masters and 300 dpi PNG fallbacks. The Makefile compiles
the editable conceptual TikZ diagrams to the same output directory. The hardware-bridge
<code>g6_delivered_waveforms.json</code> file is a derived, sanitized artifact produced
with Pulser's nominal and modulated sequence sampler; Pulser is not required to redraw it.
The audit builder reads saved pulse arrays and the two cited Figshare CSV files, performs
independent coherent/Lindblad propagation, and writes sanitized JSON. It never invokes
an optimizer. The original experimental sources remain read-only.
