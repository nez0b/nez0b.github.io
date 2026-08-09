---
layout: distill
title: "4 · The Best Coherent Pulse Is Not Always the Best Physical Pulse"
description: Lindblad dynamics, ensemble robustness, Rydberg exposure, and a reversal of the closed-system ranking
img: assets/img/neutral-atom-control/part4-noise-exposure-ranking.png
permalink: /projects/neutral-atom-control/part-4-noise-robustness/
tags: quantum-control Lindblad robustness noise
giscus_comments: false
importance: 99
category: work
show_on_projects: false
series: neutral-atom-control
series_part: 4
series_previous_url: /projects/neutral-atom-control/part-3-collocation-piccolo/
series_previous_label: "Part 3"
series_next_url: /projects/neutral-atom-control/part-5-hardware-bridge/
series_next_label: "Part 5"
status: draft
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: neutral-atom-control.bib
toc:
  - name: Three meanings of error
  - name: Open-system dynamics
  - name: The exposure budget
  - name: Open-system and ensemble optimization
  - name: The ranking reversal
  - name: The method landscape
  - name: Laboratory note
---

{% include neutral_atom_control/series_nav.liquid %}

## Three meanings of error

By the end of Part 3 we have several high-fidelity pulses, but “high fidelity” still means
“under the Hamiltonian used by the optimizer.” This chapter separates three errors that are easy to
collapse into one word:

1. **Coherent control error**: the unitary misses the desired CZ because of duration,
   leakage, discretization, or a suboptimal waveform.
2. **Dissipative error**: Rydberg decay and dephasing make the evolution nonunitary even
   when the waveform is exactly calibrated.
3. **Model uncertainty**: the experiment implements a slightly different Rabi rate,
   detuning, interaction, or transfer function from the nominal model.

A pulse can be excellent against one and poor against another. Robust control does not
mean adding one universal “robustness term”; it means deciding which uncertainty set or
noise channel matters and optimizing an appropriate aggregate.

## Open-system dynamics

The open-system comparison uses a Lindblad master equation
<d-cite key="lindblad1976generators"></d-cite>,

$$
\dot\rho=-i[H(t),\rho]
+\sum_j\left(L_j\rho L_j^\dagger
-\frac12\{L_j^\dagger L_j,\rho\}\right).
$$

For each atom, the model includes Rydberg relaxation
$\lvert r\rangle\rightarrow\lvert1\rangle$ at $0.01\,\mu\mathrm s^{-1}$ and Rydberg dephasing
at $0.05\,\mu\mathrm s^{-1}$. These are the Pulser simulation defaults used by the
simulator configuration, not fitted constants for a particular machine.

Writing $n_r=\lvert r\rangle\langle r\rvert$, the four collapse operators are

$$
\begin{aligned}
L_{1,\mathrm{rel}}&=\sqrt{\gamma_1}\,|1\rangle\langle r|\otimes I,&
L_{2,\mathrm{rel}}&=\sqrt{\gamma_1}\,I\otimes|1\rangle\langle r|,\\
L_{1,\phi}&=\sqrt{\gamma_\phi}\,n_r\otimes I,&
L_{2,\phi}&=\sqrt{\gamma_\phi}\,I\otimes n_r,
\end{aligned}
$$

with $\gamma_1=0.01\,\mu\mathrm s^{-1}$ and
$\gamma_\phi=0.05\,\mu\mathrm s^{-1}$. Relaxation changes population; the projector
collapse operator damps coherences involving $\lvert r\rangle$ without directly removing
Rydberg population. The Liouvillian is exponentiated on every saved time slice. Before
using it, the audit checks trace preservation on the full nine-level space, checks the
zero-collapse limit against coherent propagation, and verifies a representative
coherence against an independent master-equation integrator.

### Reading the collapse operators physically

The relaxation operator $\sqrt{\gamma_1}\lvert1\rangle\langle r\rvert$ has a simple
direction: it removes amplitude from the Rydberg level and deposits population in the
addressed logical level. In a quantum-trajectory picture, a jump reveals that the atom
was in $\lvert r\rangle$ and destroys the phase relation with branches that did not jump.
The master equation averages over observed and unobserved jump histories, producing both
population transfer and loss of coherence.

The dephasing operator $\sqrt{\gamma_\phi}n_r$ is different. Because it is diagonal, it
does not directly move population between levels. It damps an off-diagonal element such
as $\lvert1\rangle\langle r\rvert$, which encodes phase coherence between logical and
Rydberg components. A global gate that temporarily places two computational branches at
different Rydberg populations is therefore vulnerable even if both branches return at
the endpoint.

Tensoring each single-atom operator with the identity produces independent local noise.
This model does not include correlated laser noise, collective decay, black-body transfer
to other Rydberg states, or loss outside the three-level basis. Nor does a quoted rate by
itself specify the physical fidelity: the waveform determines how much population and
coherence are exposed to that rate over time. The following comparison holds rates fixed
and lets the trajectories supply that missing exposure information.

The metric also changes. Let $\mathcal E$ be the full channel and $P$ the computational
projector. Projecting the output back to the four-dimensional computational space gives
a generally trace-decreasing map when leakage is present. For target $CZ_\theta$, define
$F_{\mathrm{pro}}$ from the sixteen computational matrix units and define survival

$$
s=\frac{1}{4}\operatorname{tr}[P\mathcal E(P)].
$$

The unconditional Haar-average fidelity is then

$$
F_{\mathrm{avg}}^{\mathrm{uncond}}
=\frac{4F_{\mathrm{pro}}+s}{5},
$$

not $(4F_{\mathrm{pro}}+1)/5$ unless the projected process is trace preserving. The
original numerical analysis used the latter legacy convention. This revision preserves that value for
provenance and independently propagates every saved pulse to record
$F_{\mathrm{pro}}$, $s$, and the corrected unconditional score. The target phase
$\theta$ is optimized in the same CZ coset for every method.

There is also a conditional question—fidelity given that the population survived—but it
answers a different experimental protocol and is not used for the ranking below. A
single reported “average fidelity” is therefore incomplete unless it states the target,
projection, survival convention, and whether the result is conditional.

The matrix-unit construction makes the bookkeeping concrete. For computational basis
operators $E_{ij}=\lvert i\rangle\langle j\rvert$ and target $U_\theta=CZ_\theta$,

$$
\begin{aligned}
F_{\mathrm{pro}}(\theta)=\frac{1}{16}
\sum_{i,j=0}^{3}
\operatorname{tr}\!\bigl[
&(U_\theta E_{ij}U_\theta^\dagger)^\dagger\\
&\times P\mathcal E(E_{ij})P
\bigr].
\end{aligned}
$$

The scorer maximizes this expression over $\theta$. Survival uses the four diagonal
$E_{ii}$ inputs and measures their average retained trace. For a noiseless but
leakage-bearing unitary, the projected map is still trace decreasing, so the survival
correction is not solely a Lindblad issue.

Numerically, the full Liouville superoperator acts on an 81-component vectorized density
matrix. The audit fixes column-major vectorization and checks
$\mathrm{vec}(A\rho B)=(B^\mathsf T\otimes A)\mathrm{vec}(\rho)$. A transpose in the
wrong place can preserve trace and still corrupt phase-sensitive process fidelity. This
is why trace preservation, the coherent limit, and a representative off-diagonal
operator are tested separately.

### Why survival appears in average fidelity

For a trace-preserving channel on a $d$-dimensional system, the familiar relation
$F_{\mathrm{avg}}=(dF_{\mathrm{pro}}+1)/(d+1)$ contains a one because every input leaves
unit trace in the scored space. Projection changes that premise. If ten percent of an
input ends in a Rydberg state, the projected computational output has trace 0.9. Giving
the missing branch an implicit perfect score by retaining the one would overestimate the
unconditional gate quality.

The correction replaces that one with average survival $s$. For $d=4$, imagine a
projected process that performs the target perfectly whenever it survives but retains
only $s=0.9$ on average. Its process overlap scales with the surviving part, and the
unconditional expression counts loss as failure. A conditional fidelity could divide
by survival and answer “how correct was the output among retained events?” That can be
appropriate for a heralded protocol, but an ordinary deterministic gate cannot silently
postselect leakage away.

Process-map validation also needs more than propagating four populations. The diagonal
matrix units test how basis populations move; the off-diagonal $E_{ij}$ carry relative
phase and coherence. A map can preserve every computational population and still dephase
the superpositions needed for a gate. Propagating all sixteen matrix units reconstructs
the projected linear map without assuming it remains unitary.

Several numerical checks catch complementary mistakes. Hermiticity preservation tests
whether Hermitian inputs stay Hermitian. Complete positivity can be inspected through the
Choi matrix, allowing small negative eigenvalues only at integration tolerance. Full-space
trace preservation checks the Lindblad implementation, while projected trace loss is a
physical leakage signal rather than a failure. Finally, the zero-rate limit must agree
with unitary propagation under the same samples and time convention.

## The exposure budget

Dissipation acts mainly while population occupies a Rydberg level. Define the integrated
Rydberg exposure

$$
\mathcal E_r=\int_0^T
\operatorname{tr}\!\left[\rho(t)(n_r^{(1)}+n_r^{(2)})\right]dt.
$$

Across the saved pulse family, the observed noisy infidelity is well summarized by

$$
1-F_{\mathrm{avg}}^{\mathrm{uncond}}
\approx (1-F_{\mathrm{coh}})+\kappa\mathcal E_r.
$$

This is an empirical law for the selected model and pulses, not a new universal
decoherence formula. Its usefulness is conceptual: coherent accuracy and Rydberg
exposure are two budgets. Improving one while spending much more of the other can make
the physical answer worse.

The fitted $\kappa$ in the revised figure is a descriptive slope after subtracting each
pulse's independently propagated coherent trace error. It is not either collapse rate.
Different trajectories distribute exposure between one- and two-Rydberg states and
between populations and coherences, so equal $\mathcal E_r$ need not imply equal loss.
The near-linearity is useful for intuition and screening; the full master equation is
the score used for ranking.

### Exposure as an accounting exercise

If one atom carried Rydberg population 0.5 for 200 ns and otherwise stayed in the
computational space, its contribution to exposure would be
$0.5\times0.2\,\mu\mathrm s=0.1\,\mu\mathrm s$. If both atoms simultaneously carried
population 0.5 for the same interval, the definition would count
$0.2\,\mu\mathrm s$. A trajectory may reach the same endpoint with a short, highly
excited excursion or a longer, weakly excited excursion; exposure integrates those
choices into one first-order budget.

For small rates, the probability of a relaxation event is approximately the rate times
the relevant integrated population. This perturbative intuition motivates a linear
trend. Dephasing depends on coherences as well as populations, double excitation changes
the weighting, and finite rates modify the trajectory itself, so the exact answer is not
simply $(\gamma_1+\gamma_\phi)\mathcal E_r$. The fitted $\kappa$ absorbs those differences
only for this pulse family and noise model.

Exposure is therefore a useful design diagnostic but not a replacement objective by
itself. Driving exposure to zero would forbid the Rydberg mechanism that creates the
entangling phase. The meaningful question is whether the pulse accumulates the required
phase efficiently: how much useful nonlocal action is obtained per unit of dissipative
exposure, subject to coherent fidelity and control constraints.

{% include figure.liquid path="assets/img/neutral-atom-control/part4-noise-exposure-ranking.png" alt="Scatter plot of dissipative excess infidelity against Rydberg exposure and leakage-aware ranking of seven saved pulse families" caption="Left: after subtracting each independently propagated coherent trace error, a through-origin exposure fit summarizes the selected Lindblad results. Right: all seven bars use the corrected unconditional average fidelity, include open-system and robust GRAPE, and match the table below." %}

The GRAPE pulse is the clearest example. Its saved coherent error is below numerical
reporting precision, but it accumulates $0.464\,\mu\mathrm s$ of Rydberg exposure and
has corrected infidelity $1.868\times10^{-2}$. The legacy score was
$1.625\times10^{-2}$. CRAB carries only $0.245\,\mu\mathrm s$ exposure; its corrected
infidelity is $1.083\times10^{-2}$ rather than the legacy $9.38\times10^{-3}$.

The constrained $F\ge0.9999$ collocation pulse lies between them: coherent error
$8.97\times10^{-5}$, exposure $0.328\,\mu\mathrm s$, and corrected noisy infidelity
$1.411\times10^{-2}$. Hardware smoothness and decoherence efficiency are related, but
they are not the same objective.

## Open-system and ensemble optimization

There are two direct responses.

**Open-system GRAPE** differentiates through the density-matrix propagator and optimizes
the noisy objective itself. The open-system GRAPE result at $1\,\mu\mathrm s$ has coherent infidelity
$1.43\times10^{-5}$, exposure $0.254\,\mu\mathrm s$, and corrected noisy infidelity
$1.100\times10^{-2}$. It learns to avoid expensive Rydberg occupation while retaining
coherent accuracy. Its noisy score is close to CRAB's shorter pulse; the small difference
is less important than their shared improvement over coherent-only GRAPE.

**Ensemble GRAPE** keeps unitary dynamics but optimizes the mean over an uncertainty
grid. Here the grid is a $\pm2\%$ amplitude scale error:

$$
J_{\mathrm{ens}}(u)=\frac{1}{5}\sum_{\epsilon\in
\{-0.02,-0.01,0,0.01,0.02\}}
\left[1-F\bigl((1+\epsilon)u\bigr)\right].
$$

The recorded mean coherent infidelity is $1.03\times10^{-4}$. This number measures
amplitude robustness, not Lindblad noise. To compare pulse families consistently, the
saved robust pulse is nevertheless propagated afterward through the same Lindblad map;
that evaluation is not an open-system reoptimization. An ensemble can also include detuning offsets, interaction
uncertainty, or a small set of transfer functions, but the cost grows with every member.

{% include figure.liquid path="assets/img/neutral-atom-control/part4-amplitude-error-response.png" alt="Log-scale coherent trace infidelity versus multiplicative amplitude error for nominal and ensemble-robust GRAPE pulses" caption="Both curves are independently propagated from the saved arrays. The robust pulse sacrifices the extremely sharp nominal optimum to flatten the response over its $\pm2\%$ training range. This addresses parameter uncertainty, not stochastic Lindblad noise." %}

The curve is more informative than a mean. Nominal GRAPE has a very deep and narrow
minimum at zero calibration error. The ensemble pulse buys a flatter basin by accepting
a higher nominal error. Whether that trade is favorable depends on the actual amplitude
distribution and on errors not represented by a single scale factor.

The horizontal axis represents a coherent multiplicative bias applied to the complete
drive, not random sample-by-sample noise. At $+2\%$, both quadratures are scaled together,
so the optical phase is preserved while rotation rates and pulse areas change. A separate
quadrature imbalance would distort phase; an additive offset would affect weak and strong
segments differently. Robustness to one axis should not be assumed along those untested
directions.

Response curves also reveal asymmetry. A pulse need not react equally to positive and
negative scale errors because detuning, finite blockade, and time ordering make the
dynamics nonlinear. Training on a symmetric grid can still produce an asymmetric curve.
If calibration data show a biased distribution, centering or weighting the ensemble on
that distribution is more relevant than enforcing visual symmetry.

Outside the training interval, the robust curve is an extrapolation, not a guarantee.
An optimizer can flatten the specified region by moving sensitivity to its edges. A
responsible audit therefore plots a somewhat wider range than the training grid and checks
for sharp deterioration. It also evaluates nominal coherent fidelity and exposure, since
a flatter calibration response can be purchased with a longer or more dissipative path.

These two strategies answer different questions:

| Strategy                | Optimizes against                       | Does not automatically cover     |
| ----------------------- | --------------------------------------- | -------------------------------- |
| open-system GRAPE       | specified relaxation/dephasing model    | calibration and model bias       |
| ensemble GRAPE          | selected parameter distribution         | Markovian decay outside ensemble |
| constrained collocation | explicit path and hardware inequalities | unmodeled stochastic noise       |

A realistic design may combine all three: an ensemble of open-system trajectories inside
a constrained transcription. That is precisely the kind of formulation for which a
trajectory-optimization framework becomes more than convenience.

The computational price follows from the number of trajectories. Closed-system GRAPE
may propagate several basis columns. A channel objective propagates a basis of operators
in a larger Liouville space. An ensemble repeats those propagations for every calibration
member. Combining five amplitude errors with an open-system gate score can therefore
multiply an already larger derivative calculation. Parallel propagation, symmetry
reduction, and sparse trajectory abstractions become practical necessities rather than
cosmetic software features.

The aggregate objective also encodes a risk preference. Minimizing the ensemble mean can
sacrifice one edge member if the others improve. Minimizing the worst member creates a
nonsmooth minimax problem but protects the stated range more evenly. Weighting members by
a calibrated probability distribution emphasizes likely errors, while a uniform grid
claims only bounded uncertainty. A “robust pulse” is incomplete terminology unless the
ensemble and aggregate are reported.

Open-system gradients retain GRAPE's forward/backward architecture, but the generator
is the Liouvillian. For an interval $S_k=\exp(\mathcal L_k\Delta t)$, the derivative is
again a Fréchet derivative $DS_k[D\mathcal L_k]$. The dimension grows from a
9-component ket to an 81-component vectorized operator, and a gate objective samples
multiple input operators. Reduced representations and exact derivatives therefore
matter even more in open-system control.

Parameter uncertainty is conceptually different. A fixed unknown amplitude scale
$\epsilon$ produces a unitary member $U_\epsilon(T)$; an ensemble objective averages
or minimizes fidelity over several members. Lindblad dephasing is not equivalent to
drawing one static detuning error per experiment. One describes a Markovian channel,
the other epistemic or slowly varying model mismatch. They can coexist, but combining
them is justified only when calibration data support both models.

Concrete time scales make the distinction clearer. If the Rabi calibration drifts slowly
and remains nearly constant during each 1 µs gate, an ensemble of fixed scale factors is
reasonable. If the optical phase fluctuates rapidly during a gate with a short
correlation time, a Markovian dephasing channel may be a useful approximation. If it
fluctuates slowly but randomly between shots, a quasi-static detuning ensemble is closer.
Two models can produce similar decay curves in one calibration experiment yet predict
different responses to shaped control, so robustness should follow measured noise
spectra where possible.

## The ranking reversal

Under the common noise evaluator, the leading recorded families are:

| Pulse              | Survival $s$ | Legacy $1-F$ | Corrected $1-F_{\mathrm{avg}}^{\mathrm{uncond}}$ |
| ------------------ | -----------: | -----------: | -----------------------------------------------: |
| CRAB               |     0.992739 |      0.00938 |                                          0.01083 |
| open-system GRAPE  |     0.992493 |      0.00949 |                                          0.01100 |
| collocation 0.9999 |     0.990528 |      0.01222 |                                          0.01411 |
| robust GRAPE       |     0.989364 |          n/a |                                          0.01464 |
| coherent GRAPE     |     0.987827 |      0.01625 |                                          0.01868 |
| Krotov             |     0.987733 |      0.01779 |                                          0.02024 |
| Levine–Pichler     |     0.991180 |      0.02384 |                                          0.02560 |

This table is the strongest argument against optimizing one scalar in isolation.
Coherent GRAPE “wins” Part 2 and loses to four alternatives here. The optimizer did not
make a mistake; it was asked the wrong physical question.

Read the table in three passes. First compare survival. Coherent GRAPE and Krotov lose
more projected trace than the shorter CRAB pulse, consistent with their larger exposure.
Second compare the legacy and corrected columns. Their difference is not a change in the
propagation; it is the cost of no longer pretending that leaked probability survived
perfectly. Third compare methods whose corrected scores are close. CRAB and open-system
GRAPE differ by less than two ten-thousandths in this model, far below the level at which
generic rates justify a universal winner.

The ranking is conditional on the selected evaluator. Changing decay rates, adding laser
noise, or filtering the controls can reorder it again. A useful decision process therefore
keeps a Pareto set rather than one champion: coherent accuracy, duration, exposure,
roughness, uncertainty response, and hardware feasibility. Candidates can then be
rescored as the physical model improves without rerunning every search immediately.

This perspective also changes how to value an optimizer's extra decimal. Once dissipative
error is near one percent, reducing coherent error from $10^{-5}$ to $10^{-10}$ has little
effect unless the change also reduces exposure or duration. Effort is better spent on a
control objective aligned with the dominant error budget, or on experiments that identify
which budget is actually dominant.

A practical selection loop begins with a coherent Pareto set, rescoring every candidate
under the same channel and uncertainty models. Clearly dominated pulses can be dropped,
but close candidates should survive until calibration improves. If two scores differ by
less than the uncertainty in the rates or interaction strength, choosing the easier pulse
to synthesize may be more defensible than declaring a numerical winner. The evaluator,
its parameters, and its uncertainty are therefore part of the result.

This is where control theory meets experimental design. Sensitivity calculations identify
which calibration parameter most changes the ranking; a targeted measurement of that
parameter can be more valuable than another optimization run. Optimal control is not only
the production of waveforms. It is a framework for deciding which model refinement or
measurement would most improve the next waveform decision.

## The method landscape

The claim that noise reorders the ranking is a strong one, and it sits in a literature with several distinct strategies for taking noise seriously during optimization rather than after it.

<table class="nac-metric-table">
<thead><tr><th>Strategy</th><th>Noise model</th><th>Mechanism</th><th>Demonstrated on Rydberg?</th></tr></thead>
<tbody>
<tr><td>Post-hoc evaluation (this chapter)</td><td>Lindblad dissipation <d-cite key="lindblad1976generators"></d-cite></td><td>Optimize closed, score open</td><td>Yes, here</td></tr>
<tr><td>Ensemble / robust GRAPE <d-cite key="kobzar2008broadband"></d-cite></td><td>Parameter spread</td><td>Average fidelity over a distribution</td><td>Originated in NMR</td></tr>
<tr><td>Open-system GRAPE <d-cite key="schulteherbruggen2011opengrape"></d-cite></td><td>Dissipation</td><td>Propagate the density matrix directly</td><td>General</td></tr>
<tr><td>Open-system Krotov <d-cite key="goerz2014krotovopen"></d-cite></td><td>Dissipation</td><td>Monotonic update on open dynamics</td><td>General</td></tr>
<tr><td>Robust Rydberg co-optimization <d-cite key="mohan2023robust"></d-cite></td><td>Both</td><td>Pulse and Rydberg level chosen together</td><td>Yes</td></tr>
</tbody>
</table>

The two mechanisms are genuinely different and are often conflated. Ensemble methods average the closed-system fidelity over a distribution of Hamiltonian parameters, which addresses _miscalibration_ — the Rabi frequency is not quite what you asked for. The trick originates in NMR, where RF inhomogeneity across a sample poses exactly this problem <d-cite key="kobzar2008broadband"></d-cite>, and it maps directly onto detuning and amplitude errors in a tweezer array. Open-system methods instead propagate the density matrix under a dissipator and address _decoherence_ — population genuinely leaves the computational space. A pulse can be robust in the first sense and bad in the second.

That distinction is what makes the exposure budget in this chapter the right diagnostic: it measures time spent in $\lvert r\rangle$, which is the quantity the decay channels actually charge for. The underlying rates — including blackbody-induced transitions out of the Rydberg manifold, which dominate at room temperature for the levels used here — are tabulated in the atomic-physics literature <d-cite key="beterov2009blackbody"></d-cite>.

<div class="nac-callout">
<strong>Scope of the reversal.</strong> The ranking reversal reported here is a statement about these pulses, this Hamiltonian, and this noise model. It is evidence that coherent-fidelity ranking is not automatically the physical ranking — not evidence that any particular method is generally more robust.
</div>

## Laboratory note

<div class="nac-lab-note"><strong>What changed.</strong> Adding realistic-enough decay and dephasing did more than lower every fidelity by a constant. It reordered the pulse family. The short, structured CRAB pulse became competitive with an open-system gradient, while the closed-system champion paid for nearly twice the Rydberg exposure.</div>

This still is not a hardware prediction. The rates are a generic model, the waveform is
not yet sampled on a device clock, and state preparation and measurement are absent.
Part 5 crosses that boundary—and changes from the two-qutrit CZ task to a Bell-state
hardware surrogate rather than pretending the available analog interface implements the
same gate.

That handoff preserves the hierarchy of evidence. The open-system scorer asks what the
saved mathematical controls do under one dissipative model. It does not establish that a
control channel can emit them, that a calibrated register realizes the assumed $V$, or
that measured counts resolve the modeled process fidelity. Each of those questions needs
its own representation and validation step, which is why the final chapter begins from
arrays rather than calling them waveforms.
That vocabulary keeps numerical predictions useful without promoting them beyond the
evidence actually supplied by the model.
It also prevents false precision.

{% include neutral_atom_control/series_nav.liquid %}
