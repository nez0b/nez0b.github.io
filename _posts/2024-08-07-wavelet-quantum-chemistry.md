---
layout: distill
title: "From Daubechies wavelets to qubits: an N₂ quantum-chemistry pipeline"
description: How an adaptive wavelet representation becomes a validated active-space Hamiltonian, with a fair N₂ benchmark, VQE, QSCI, and SQD.
tags: quantum-chemistry wavelets quantum-computing
categories: scientific-computing
giscus_comments: false
date: 2024-08-07
featured: true
related_posts: true
bibliography: 2024-08-wavelet-qc.bib
og_image: https://nez0b.github.io/assets/img/wavelet_qc/n2_dissociation_headline.png

authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research
      url: "https://github.com/nez0b/bigdft-drug-design"

toc:
  - name: The central question
  - name: Wavelets in one picture
    subsections:
      - name: Multiresolution instead of atom-centered functions
      - name: The three practical controls
  - name: From orbitals to qubits
    subsections:
      - name: Turning four-index integrals into Poisson solves
      - name: A validation ladder
  - name: Making the N₂ comparison fair
    subsections:
      - name: Why frozen core matters
      - name: The completed result
  - name: What resolution costs
  - name: Adding QSCI and SQD
  - name: Where the current pipeline stops
  - name: What this result does and does not say
  - name: Read and reproduce the code

_styles: >
  .wavelet-lede {
    font-size: 1.08rem;
    line-height: 1.75;
  }
  .wavelet-pipeline {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.65rem;
    margin: 1.5rem 0 2rem;
  }
  .wavelet-pipeline > div {
    border: 1px solid var(--global-divider-color);
    border-radius: 0.45rem;
    padding: 0.85rem;
    text-align: center;
    background: var(--global-card-bg-color);
  }
  .wavelet-pipeline strong {
    color: var(--global-theme-color);
    display: block;
    margin-bottom: 0.25rem;
  }
  .wavelet-callout {
    border-left: 4px solid var(--global-theme-color);
    background: var(--global-card-bg-color);
    margin: 1.4rem 0;
    padding: 0.9rem 1.1rem;
  }
  .wavelet-caution {
    border-left-color: #b24a3a;
  }
  .wavelet-equation {
    margin: 1rem 0;
    overflow-x: auto;
    text-align: center;
  }
  d-article table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
  }
---

<p class="wavelet-lede">
Most quantum-chemistry workflows begin with atom-centered Gaussian orbitals. BigDFT offers a different starting point: compactly supported Daubechies wavelets on an adaptive real-space grid. This article follows that representation all the way from molecular orbitals to a qubit Hamiltonian and asks whether the resulting N₂ bond curve can be compared fairly with a conventional Gaussian-basis calculation.
</p>

N₂ is a useful test case because it is small enough for exact active-space checks but demanding enough to expose basis resolution, frozen-core conventions, bond breaking, and solver error. The workflow extracts one- and two-electron integrals from wavelet orbitals, validates every representation boundary, and connects the same Hamiltonian to full configuration interaction (FCI), variational quantum eigensolver (VQE), quantum-selected configuration interaction (QSCI), and sample-based quantum diagonalization (SQD).

The final workflow is compact enough to state in one line:

<div class="wavelet-pipeline">
  <div><strong>BigDFT</strong>adaptive wavelet orbitals</div>
  <div><strong>Integral factory</strong>$h_{pq}$ and $(pq|rs)$</div>
  <div><strong>Active space</strong>freeze and select orbitals</div>
  <div><strong>Qubit map</strong>Jordan–Wigner Hamiltonian</div>
  <div><strong>Solvers</strong>FCI, VQE, QSCI, SQD</div>
</div>

The complete implementation, datasets, analysis scripts, tests, and machine-readable results live in the [`agent/wavelet-qc-macos-revival` branch](https://github.com/nez0b/bigdft-drug-design/tree/agent/wavelet-qc-macos-revival) of the GitHub codebase. Readers who want implementation or environment details should follow the README and documentation there.

## The central question

The motivation is not that wavelets are automatically “better” than Gaussian orbitals. It is more specific.

Quantum chemistry must make two truncations. First, a classical electronic-structure calculation represents continuous orbitals in a finite numerical basis. Second, a quantum algorithm keeps a finite set of spatial orbitals, usually turning $m$ spatial orbitals into $2m$ spin-orbital qubits. A representation that gives controlled real-space accuracy while keeping the chemically useful orbital space compact could therefore help both sides of the pipeline.

BigDFT is attractive because it uses compactly supported Daubechies wavelets on an adaptive grid rather than a fixed list of atom-centered Gaussian functions <d-cite key="daubechies1988,genovese2008"></d-cite>. Prior work showed that wavelet molecular orbitals can be useful inputs to quantum computations and vibrational calculations <d-cite key="hong2022,chou2023"></d-cite>. The question here is whether that idea can support a reproducible end-to-end integral pipeline and a scientifically controlled molecular benchmark.

The standard is therefore stronger than obtaining a plausible energy: the calculation needs a complete potential-energy curve, a properly matched reference, and independent checks at every representation boundary.

## Wavelets in one picture

### Multiresolution instead of atom-centered functions

A molecular orbital is still an expansion,

<div class="wavelet-equation" markdown="1">

$$
\phi_i(\mathbf r)=\sum_\alpha c_{i\alpha}\,\Phi_\alpha(\mathbf r),
$$

</div>

but the basis functions $\Phi_\alpha$ now live on a real-space grid. Scaling functions describe the smooth part of an orbital; wavelets add localized detail. Both have compact support, so a basis function is exactly zero outside a finite interval.

{% include figure.liquid path="/assets/img/wavelet_qc/wavelet_basis_1d.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A Daubechies scaling function and wavelet with compact support, shown relative to the nitrogen bond length" caption="A one-dimensional view of the scaling function and wavelet used to illustrate the basis. Compact support and localized detail are the important ideas; this is not a molecular-orbital plot." zoomable=true %}

The three-dimensional basis is made from tensor products. At a coarse grid point there is one $\phi\phi\phi$ scaling function. In the fine region, the seven other $\phi/\psi$ combinations add detail. This is why the coefficient count in our cost analysis is naturally written as

<div class="wavelet-equation" markdown="1">

$$
N_\text{basis}=N_\text{coarse}+7N_\text{fine}.
$$

</div>

The useful mental model is a microscope: retain a broad, inexpensive description of the vacuum and smooth orbital tails, then add resolution near the atoms where the functions vary rapidly.

### The three practical controls

Wavelets do not remove convergence choices. They make those choices geometric and systematic:

- `hgrid` is the real-space spacing. Smaller values resolve shorter-length-scale features and cost more.
- `crmult` controls the radius of the coarse region around each atom.
- `frmult` controls the smaller fine-resolution region near each nucleus.

{% include figure.liquid path="/assets/img/wavelet_qc/wavelet_regions.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Schematic coarse and fine wavelet grids surrounding two nitrogen atoms" caption="Schematic two-dimensional slices through the N₂ support regions. Blue points carry coarse scaling functions; red points also carry the seven fine wavelets. The circles are support masks, not electron-density contours." zoomable=true %}

This is an important correction to the loose phrase “basis-set free.” There is no cc-pVDZ-style catalog to choose, but there is still a finite basis. Its error is controlled by grid spacing and spatial support.

## From orbitals to qubits

Once BigDFT has converged the occupied and requested virtual orbitals, the downstream problem is familiar. For orthonormal orbitals, the electronic Hamiltonian is

<div class="wavelet-equation" markdown="1">

$$
H=\sum_{pq} h_{pq}a_p^\dagger a_q
 +\frac{1}{2}\sum_{pqrs}(pq|rs)a_p^\dagger a_r^\dagger a_s a_q
 +E_\text{core}.
$$

</div>

The custom BigDFT postprocessor writes symmetry-unique `hpq` and `hpqrs` records. The maintained Python loader restores the matrix, electron-repulsion, and spin symmetries, reduces the problem to a chosen active space, and maps the fermionic operators to Pauli strings with Jordan–Wigner.

### Turning four-index integrals into Poisson solves

The expensive object is

<div class="wavelet-equation" markdown="1">

$$
(pq|rs)=\iint
\phi_p(\mathbf r_1)\phi_q(\mathbf r_1)
\frac{1}{|\mathbf r_1-\mathbf r_2|}
\phi_r(\mathbf r_2)\phi_s(\mathbf r_2)
\,d\mathbf r_1d\mathbf r_2.
$$

</div>

Rather than introduce unrelated quadrature machinery, the extractor reuses BigDFT’s real-space Poisson solver. For each orbital pair it forms $\rho_{pq}(\mathbf r)=\phi_p(\mathbf r)\phi_q(\mathbf r)$, solves

<div class="wavelet-equation" markdown="1">

$$
\nabla^2V_{pq}(\mathbf r)=-4\pi\rho_{pq}(\mathbf r),
$$

</div>

and evaluates $\int \rho_{rs}(\mathbf r)V_{pq}(\mathbf r)d\mathbf r$. This is the central bridge from the wavelet representation to an ordinary second-quantized chemistry Hamiltonian.

### A validation ladder

A plausible energy is not enough. The rebuilt pipeline checks each handoff independently:

1. The equilibrium fixture contains all 240 expected one-electron records and 7,260 symmetry-unique two-electron records for 15 spatial orbitals.
2. Randomly selected permutations satisfy the real-integral symmetries.
3. The occupied-orbital Hartree–Fock expression can be reconstructed from the emitted integrals. It is 0.437 Ha above the BigDFT PBE energy on the same orbitals, as expected from the different exchange-correlation treatment—not from a missing integral.
4. For an MP2-guided CAS(4e,4o), exact diagonalization of the explicit eight-qubit Jordan–Wigner operator agrees with an independent PySCF FCI calculation to $6.75\times10^{-14}$ Ha.
5. The eight-qubit Hamiltonian contains 185 Pauli terms. A small 64-parameter hardware-efficient VQE lands 4.05 mHa above exact: useful as a runnable demonstration, but outside the conventional 1.6 mHa “chemical accuracy” line.

Only after those checks do I interpret the molecular curve.

## Making the N₂ comparison fair

### Why frozen core matters

The most subtle part of the benchmark is not fitting the curve. It is deciding what can be compared.

The BigDFT calculation uses an HGH-K/PBE pseudopotential and explicitly represents ten valence electrons. A normal all-electron N₂ calculation represents fourteen electrons. Comparing the two absolute totals—roughly $-20$ Ha and $-109$ Ha—would mostly compare different core conventions and energy zeros.

I therefore used two PySCF references:

1. A valence-matched GTH-PBE/`gth-dzvp` calculation with ten explicit electrons.
2. An all-electron cc-pVDZ calculation with fourteen total electrons, but with the two doubly occupied N 1s-derived molecular orbitals frozen. That leaves the same ten correlated valence electrons in CAS(10e,10o).

Freezing the core does not make the absolute all-electron energy equal to a pseudopotential energy. It makes the _correlated valence problem_ comparable. Each potential curve is then shifted to its own minimum,

<div class="wavelet-equation" markdown="1">

$$
\Delta E_m(R)=E_m(R)-\min_R E_m(R),
$$

</div>

so the comparison uses quantities that survive a change of energy zero: equilibrium distance, local curvature, harmonic frequency, and nearby curve shape.

<div class="wavelet-callout wavelet-caution">
<strong>What this comparison is not:</strong> GTH-PBE is not the exact same pseudopotential as BigDFT’s HGH-K/PBE, and the PySCF helper starts from RHF orbitals while BigDFT supplies PBE Kohn–Sham orbitals. The plot is a controlled cross-method benchmark, not a pure isolation of “wavelets versus Gaussians.”
</div>

### The completed result

{% include figure.liquid path="/assets/img/wavelet_qc/n2_dissociation_headline.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Minimum-shifted N2 dissociation curves for BigDFT wavelets, a valence pseudopotential Gaussian calculation, and an all-electron frozen-core calculation" caption="Three CAS(10e,10o) N₂ curves, each shifted to its own minimum. Vertical lines mark the fitted equilibrium distances; the table reports zero-independent observables." zoomable=true %}

| Method                       |   Electrons represented | Correlated space | $r_e$ (Å) | $\omega_e$ (cm⁻¹) |
| ---------------------------- | ----------------------: | ---------------: | --------: | ----------------: |
| BigDFT wavelet / HGH-PBE     |              10 valence |     CAS(10e,10o) |    1.0875 |            2548.5 |
| PySCF GTH-PBE / gth-dzvp     |              10 valence |     CAS(10e,10o) |    1.1066 |            2452.5 |
| PySCF all-electron / cc-pVDZ | 14 total; 4 frozen core |     CAS(10e,10o) |    1.1050 |            2358.5 |
| Experiment                   |                       — |                — |    1.0977 |            2358.6 |

The wavelet equilibrium distance is 0.0102 Å shorter than experiment, an error of about 0.93%. The fitted frequency is about 8% high. The bond length is the stronger result: the available scan is spaced by 0.2 Å near equilibrium and retains only five virtual orbitals, while a second derivative is especially sensitive to both choices. A denser near-minimum scan is the clearest scientific next step.

The all-electron frozen-core frequency happens to match experiment closely, but this small benchmark does not support a general accuracy ranking. The important result is that the wavelet curve has a physically sensible minimum and can be compared without subtracting incompatible absolute energies.

## What resolution costs

The parameter scan is useful because it exposes both convergence and computational price.

{% include figure.liquid path="/assets/img/wavelet_qc/scan_convergence.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Energy convergence as hgrid, crmult, and frmult are varied one at a time" caption="One-parameter-at-a-time convergence at the equilibrium geometry. The lower panels show differences from the finest value in each branch; the shaded region ends at 1.6 mHa." zoomable=true %}

Decreasing `hgrid` from 0.45 to 0.20 bohr changes the BigDFT energy from $-19.905076$ to $-19.910426$ Ha and the fixed CAS(10e,10o) energy from $-19.568101$ to $-19.572608$ Ha. The `hgrid=0.35` CAS result is already within 1.6 mHa of the finest tested point.

`frmult=6,7,8` is essentially flat here. Increasing `crmult` converges the Kohn–Sham energy, but the fixed-size CAS energy is not monotonic. That is not a violation of the variational principle: changing the box can change which diffuse virtual orbitals occupy the five retained virtual slots, so it is not the same subspace at every point.

{% include figure.liquid path="/assets/img/wavelet_qc/scan_cost.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Measured wall time and peak memory as functions of the number of wavelet coefficients" caption="Measured cost of one N₂ SCF-plus-integral point in the benchmark environment. The scatter includes different grid and support branches, so the dashed power laws are descriptive rather than universal scaling claims." zoomable=true %}

Across the parameter scan, the extractor grows from about 57 seconds and 0.5 GB RSS at `hgrid=0.45` to about 406 seconds and 2.9 GB at `hgrid=0.20`. Three complete curves at `hgrid=0.35`, 0.25, and 0.20 give fitted bond lengths of 1.0876, 1.0875, and 1.0873 Å. That stability is reassuring even though the absolute frequency remains resolution- and sampling-sensitive. These timings describe the recorded benchmark environment rather than a universal performance model.

## Adding QSCI and SQD

The next question is what happens after the wavelet Hamiltonian reaches a sample-based eigensolver.

QSCI samples a quantum state in the occupation-number basis, retains the important Slater determinants, and diagonalizes the Hamiltonian in that selected classical subspace <d-cite key="kanno2026"></d-cite>. SQD adds particle-number postselection, batches, iterative configuration recovery, and distributed diagonalization; its appeal is to move most of the energy evaluation away from repeated Pauli measurements <d-cite key="robledomoreno2024"></d-cite>.

For a controlled software test, I used the real wavelet CAS(6e,6o) Hamiltonian: 12 Jordan–Wigner qubits but only

<div class="wavelet-equation" markdown="1">

$$
\binom{6}{3}\binom{6}{3}=400
$$

</div>

determinants in the fixed $(N_\alpha,N_\beta)=(3,3)$ sector. Samples were drawn from the squared coefficients of the exact PySCF FCI vector.

| Oracle shots | Unique determinants sampled | QSCI error (mHa) | SQD error (mHa) |
| -----------: | --------------------------: | ---------------: | --------------: |
|        5,000 |                          25 |            5.561 |           5.561 |
|       20,000 |                          36 |            5.558 |           5.558 |
|      100,000 |                          40 |            1.536 |           1.536 |
|      500,000 |                          43 |            0.184 |           0.184 |

The convergence has a simple interpretation: rare but energetically important determinants appear as sample coverage grows. QSCI and SQD agree exactly here because every sample already has the correct particle number and both routes diagonalize the same set of unique configurations. Recovery has nothing extra to repair in this clean experiment.

<div class="wavelet-callout wavelet-caution">
<strong>This is not QPU data.</strong> The exact FCI distribution is a classical sampler oracle chosen to validate subspace construction, bit ordering, energy constants, and diagonalization. It says nothing about state-preparation depth, device noise, mitigation, or quantum advantage. Those are deliberately separated from the software check.
</div>

A separate QURI-QSCI diagnostic did not converge: its energy stayed 3.979 Ha from the CASCI reference and did not respond to the requested subspace size. It is recorded in the codebase as a negative diagnostic rather than presented as a successful result.

## Where the current pipeline stops

The codebase also contains data from a 109Asp calculation with 132 spatial orbitals. That is exactly where scientific restraint matters.

For real orbitals, the number of symmetry-unique two-electron records is

<div class="wavelet-equation" markdown="1">

$$
M=\frac{n(n+1)}{2},\qquad N_\text{unique}=\frac{M(M+1)}{2}.
$$

</div>

At $n=132$, this is 38,531,031 records. The complete text export is 1.387 GB and is not vendored. A separate file stops at 100,000 records—only 0.26%—and comes from a different run, so the two cannot be concatenated. Expanding the 132 spatial orbitals into a dense 264-spin-orbital `float64` tensor would require about 38.9 GB before solver intermediates.

The current workflow therefore records provenance and resource estimates only. It does **not** invent a 109Asp molecular energy from a truncated tensor. Sparse/out-of-core integrals, density fitting, localization, embedding, or a different active-space strategy are required before that system becomes a real downstream calculation.

## What this result does and does not say

What I am comfortable claiming:

- The repository implements a complete wavelet-to-integral pipeline with documented inputs and outputs.
- The emitted integrals pass completeness, symmetry, mean-field, FCI, and qubit-mapping checks.
- The N₂ CAS(10e,10o) wavelet curve now has a completed, scientifically matched comparison; its fitted equilibrium distance is within about 1% of experiment.
- The maintained QSCI/SQD classical stages converge to FCI under a transparent exact-distribution sampler oracle.

What I am **not** claiming:

- a pure wavelet-versus-Gaussian basis error measurement;
- a production-quality vibrational frequency from the coarse geometry grid;
- a successful 109Asp energy or drug-binding calculation;
- quantum advantage, quantum-hardware data, or chemical accuracy from the small VQE.

The central lesson is that a useful end-to-end demonstration depends as much on matched comparisons and explicit limitations as it does on the solver result.

## Read and reproduce the code

Implementation and installation details are intentionally kept in the repository rather than duplicated in this article. The codebase includes a locked Python environment, numbered analysis scripts, tests, source JSON, generated figures, and detailed method notes. Start with the [`agent/wavelet-qc-macos-revival` branch](https://github.com/nez0b/bigdft-drug-design/tree/agent/wavelet-qc-macos-revival). Within that codebase, the main entry points are:

- `README.md` for the overview and result table;
- `INSTALL.md` for the environment and installation guide;
- `docs/METHODOLOGY.md` for the comparison protocol;
- `RESULTS.md` for numerical results and limitations;
- `docs/SQD_QSCI.md` for the sample-based solver workflow;
- `docs/PROVENANCE.md` for data and contributor provenance.

The compact analyses start from committed integral fixtures, so readers can reproduce the figures and solver checks without first generating a new wavelet calculation. Read the repository documentation for implementation boundaries, data provenance, and contributor attribution.
