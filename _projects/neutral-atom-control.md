---
layout: distill
title: Neutral-Atom Pulse Control
description: Five experiments in turning a Rydberg Hamiltonian into a physical two-qubit pulse
permalink: /projects/neutral-atom-control/
tags: quantum-control neutral-atoms optimal-control rydberg
giscus_comments: false
img: assets/img/neutral-atom-control/cover.png
importance: 3
category: work
show_on_projects: true
series: neutral-atom-control
status: draft
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: neutral-atom-control.bib
toc:
  - name: The question
  - name: Reading map
  - name: What is being optimized
  - name: Experiment map
  - name: Conventions and provenance
---

<div class="nac-series-nav">
  <div class="nac-draft"><strong>Draft series</strong> · complete first-pass prose and figures, not yet published</div>
  <div class="nac-series-kicker">Neutral-atom pulse control · five chapters</div>
</div>

{% include figure.liquid path="assets/img/neutral-atom-control/cover.png" alt="Isometrically projected five-by-four optical-tweezer array with excitation beams, two adjacent Rydberg atoms, overlapping blockade regions, and a ground-to-Rydberg control inset" caption="A projected 5×4 optical-tweezer array. Two adjacent atoms are excited to Rydberg states; the overlapping ellipses indicate their blockade regions, the gold in-plane arrow marks their interaction, and the green beams indicate optical addressing." %}

## The question

A neutral-atom gate begins as a compact equation and ends as voltages, optical fields,
clock ticks, and counts. Between those endpoints sit several distinct optimization
problems. A pulse that is perfect for a closed Schrödinger equation may be fragile to a
one-percent calibration shift. A pulse that is smooth in an optimizer may be reshaped by
the control channel. A reduced state representation may optimize quickly while silently
forgetting a phase that makes the gate a CZ.

This series follows those failure modes rather than hiding them. The first four chapters
study a locally phase-equivalent CZ in a two-qutrit model: first as a coherent gate,
then as a constrained trajectory, and finally as an open-system process. The fifth
chapter deliberately changes the task. It translates a hardware-feasible Bell-state
pulse to Pulser and develops a proposed validation run. That Bell pulse is **not**
presented as a native hardware execution of the earlier CZ.

The intended reader knows bras, kets, matrices, and elementary quantum gates. Control
theory is built from the ground up: state and control, costate and Pontryagin's maximum
principle, forward/backward gradients, randomized bases, and direct collocation.

## Reading map

<div class="nac-chapter-grid">
  <div class="nac-chapter-card"><h3>1 · Model</h3><p>From the physical $g-e-r$ ladder to the effective $\lvert0\rangle,\lvert1\rangle,\lvert r\rangle$ Hamiltonian, blockade, CZ, and PMP.</p><a href="/projects/neutral-atom-control/part-1-foundations/">Read Part 1 →</a></div>
  <div class="nac-chapter-card"><h3>2 · Pulse search</h3><p>GRAPE, Krotov, and CRAB solve related—but not identical—finite-dimensional searches.</p><a href="/projects/neutral-atom-control/part-2-grape-krotov-crab/">Read Part 2 →</a></div>
  <div class="nac-chapter-card"><h3>3 · Trajectories</h3><p>Direct collocation and Piccolo make the trajectory and its constraints decision variables.</p><a href="/projects/neutral-atom-control/part-3-collocation-piccolo/">Read Part 3 →</a></div>
  <div class="nac-chapter-card"><h3>4 · Noise</h3><p>Lindblad dynamics and Rydberg exposure reorder the apparent winners.</p><a href="/projects/neutral-atom-control/part-4-noise-robustness/">Read Part 4 →</a></div>
  <div class="nac-chapter-card"><h3>5 · Hardware</h3><p>Clock grids, modulation, geometry, SPAM, and an explicitly unexecuted run plan.</p><a href="/projects/neutral-atom-control/part-5-hardware-bridge/">Read Part 5 →</a></div>
</div>

## What is being optimized

The control vector is

$$u(t)=\bigl(\Omega_x(t),\Omega_y(t),\Delta(t)\bigr),$$

with a hard Rabi-amplitude limit and, in the collocation experiments, endpoint, slew,
and curvature limits. The coherent CZ objective uses the computational block $M$ of
the propagated unitary and optimizes over a local single-qubit phase,

$$
F_{\mathrm{tr}}=\max_\theta\frac{\left|\operatorname{tr}
\left(CZ_\theta^\dagger M\right)\right|^2}{16},\qquad
CZ_\theta=\operatorname{diag}(1,e^{i\theta},e^{i\theta},-e^{2i\theta}).
$$

This is a squared trace-overlap convention. Part 4 reports process fidelity, survival,
and the leakage-aware unconditional average fidelity

$$
F_{\mathrm{avg}}^{\mathrm{uncond}}
=\frac{4F_{\mathrm{pro}}+s}{5},\qquad
s=\frac14\operatorname{tr}[P\mathcal E(P)].
$$

The earlier trace-preserving approximation $(4F_{\mathrm{pro}}+1)/5$ is retained only
for provenance. Part 5
reports Bell-state population/fidelity. These labels are intentionally different: the
numbers should not be placed in one leaderboard as though they were the same metric.

## Experiment map

| Chapter | Scientific question                                       | Main artifact                                              |
| ------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| 1       | What Hamiltonian and phase convention define the CZ?      | analytic baselines, blockade scan, canonical constants     |
| 2       | How do three pulse-search parameterizations behave?       | GRAPE, Krotov, CRAB, and reference pulses                  |
| 3       | What changes when states and controls are co-optimized?   | constrained CZ trajectories and Piccolo findings           |
| 4       | How do dissipation and uncertainty reorder the solutions? | Lindblad scores, exposure, open- and robust-control pulses |
| 5       | What survives translation to a device-facing sequence?    | delivered waveforms, geometry audit, anonymized run plan   |

## Conventions and provenance

Angular controls are stored in rad/µs and displayed as ordinary frequencies,
$\Omega/2\pi$ and $\Delta/2\pi$, in MHz. Durations are shown in ns when discussing
gates or hardware clocks and in µs for differential equations. Every quantitative
figure is regenerated from the saved JSON artifacts; the optimizers are not rerun.

The series is a report of a particular experimental path, not a claim that one method is
universally superior. The useful comparison is structural: which variables and
constraints a method can express, which derivatives it needs, and which physical errors
remain outside its model.
