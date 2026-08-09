---
layout: distill
title: Quantum Simulation Techniques
description: Mechanism-first guides to statevectors, tensor-network representations, and operator-based simulation
permalink: /projects/quantum-simulation/
tags: quantum-computing quantum-simulation numerical-methods
img: assets/img/quantum-simulation/cover.png
importance: 2
category: work
show_on_projects: true
series: quantum-simulation
status: draft
giscus_comments: false
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
toc:
  - name: What this series is about
  - name: Reading map
  - name: Why representations matter
  - name: Reproducibility
---

<div class="mb-4 p-3 border rounded">
  <strong>Draft series</strong> · the first chapter is ready for review; later chapters are planned, not yet published
</div>

{% include figure.liquid path="assets/img/quantum-simulation/cover.png" alt="Four compact binary indices mapped to disjoint pairs of statevector amplitudes whose indices differ at one target bit" caption="A one-qubit gate is a collection of independent two-amplitude updates. The computational challenge is to enumerate those pairs directly and without duplication." %}

## What this series is about

A quantum simulator does not merely implement the Schrödinger equation. It chooses a
representation of the quantum state, a way to find the data touched by each operation,
and a memory-access pattern that determines whether the machine spends its time doing
arithmetic or waiting for bytes.

This series studies those choices from the implementation outward. Each chapter starts
from the mathematical object being represented, derives the indexing or contraction
rule that makes an operation possible, and then asks what the computer actually has to
allocate, move, and update. The goal is not to rank all simulators with one benchmark.
Different representations win in different regimes.

The intended reader knows basic linear algebra, complex amplitudes, and quantum gates.
The implementation discussion stays close to those concepts: small worked examples,
bit diagrams, contraction patterns, and measurements with their provenance preserved.

## Reading map

### 1 · Statevectors — available now

[**How to Optimize Statevector Simulation**](/projects/quantum-simulation/how-to-optimize-statevector-simulation/)
explains the zero-bit insertion trick for applying a one-qubit gate directly to
$2^{N-1}$ disjoint amplitude pairs. Julia supplies the pedagogical path; a Rust
implementation supplies the reproducible benchmark; a merged QuEST contribution shows
why instruction-level gains shrink at the memory-bandwidth wall.

### Planned chapters

- **Matrix product states:** Schmidt rank, bond dimension, canonical forms, and where
  truncation enters the algorithm.
- **General tensor networks:** contraction order, intermediate tensors, and the tension
  between arithmetic count and memory use.
- **Pauli propagation:** evolving observables rather than amplitudes, and when operator
  growth replaces entanglement growth as the limiting quantity.
- **Hybrid techniques:** choosing or switching representations according to circuit
  structure rather than treating one simulator as universal.

These topics are a roadmap, not links to unfinished articles.

## Why representations matter

An $N$-qubit statevector stores $2^N$ complex numbers and gives exact, direct access to
every amplitude. That generality makes it an excellent baseline, but its memory doubles
with each added qubit. Tensor-network methods trade that uniform access for structure:
they can compress weakly entangled states, yet their cost can rise sharply when bond
dimensions grow. Pauli methods move the calculation into operator space and expose a
different notion of complexity.

The common theme is therefore not a particular library or language. It is the mapping
between a mathematical operation and the smallest correct movement of data.

## Reproducibility

The diagrams in this series have editable TikZ sources. Quantitative plots are generated
from saved, human-readable benchmark data rather than copied from transient benchmark
output. For the statevector chapter, the data file records the Rust commit, compiler,
hardware, command, confidence intervals, and the provenance of the supporting QuEST
measurements.

The series remains a draft until its prose, figures, links, and rendered layouts have
been reviewed together.
