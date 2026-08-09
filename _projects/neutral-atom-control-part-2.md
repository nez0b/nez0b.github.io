---
layout: distill
title: "2 · Three Traditional Routes: GRAPE, Krotov, and CRAB"
description: Forward-backward gradients, sequential updates, randomized bases, and what their pulse shapes reveal
img: assets/img/neutral-atom-control/part2-method-pulses.png
permalink: /projects/neutral-atom-control/part-2-grape-krotov-crab/
tags: quantum-control GRAPE Krotov CRAB
giscus_comments: false
importance: 99
category: work
show_on_projects: false
series: neutral-atom-control
series_part: 2
series_previous_url: /projects/neutral-atom-control/part-1-foundations/
series_previous_label: "Part 1"
series_next_url: /projects/neutral-atom-control/part-3-collocation-piccolo/
series_next_label: "Part 3"
status: draft
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: neutral-atom-control.bib
toc:
  - name: One problem, three search spaces
  - name: GRAPE
  - name: Krotov
  - name: CRAB
  - name: Reading the pulses
  - name: A fair comparison
  - name: The method landscape
  - name: Laboratory note
---

{% include neutral_atom_control/series_nav.liquid %}

## One problem, three search spaces

Part 1 gave us a finite-blockade Hamiltonian, a phase-aware CZ objective, and an analytic
baseline. Here we hold the physical model fixed and change the _language in which a pulse
is searched_. That distinction is more useful than sorting methods into “old” and “new.”
GRAPE, Krotov, and CRAB can all solve quantum-control problems, but they expose different
variables to the optimizer and handle hardware structure differently.

For the main comparison, GRAPE and Krotov use $T=1\,\mu\mathrm s$ and 100 constant
time slices. CRAB dresses the 683.2 ns analytic pulse with eight basis coefficients.
The Rabi quadratures obey a component box
$\lvert\Omega_x\rvert,\lvert\Omega_y\rvert\le\Omega_{\max}/\sqrt2$; detuning is fixed at zero for GRAPE
and Krotov, while the CRAB seed retains its constant analytic detuning. These differences
are stated because the final infidelities are **not** a controlled software benchmark.

| Method | Search variables           | Derivative              | Natural strength                   | Typical difficulty                 |
| ------ | -------------------------- | ----------------------- | ---------------------------------- | ---------------------------------- |
| GRAPE  | every time-slice amplitude | exact/discrete gradient | large flexible control space       | jagged pulses and local optima     |
| Krotov | time-local field updates   | forward/backward states | monotonic theory under assumptions | clipping can break the guarantee   |
| CRAB   | a few basis coefficients   | often derivative-free   | compact, bandwidth-aware ansatz    | misses solutions outside the basis |

## GRAPE

For piecewise-constant controls,

$$
U(T)=U_{N-1}\cdots U_1U_0,\qquad
U_k=\exp[-iH(u_k)\Delta t].
$$

Naively differentiating the whole product once per parameter would be expensive. GRAPE
stores forward products $X_k=U_{k-1}\cdots U_0$ and backward products carrying the
terminal objective. A gradient component becomes a local contraction,

$$
\frac{\partial F}{\partial u_{j,k}}
\propto \operatorname{Re}\operatorname{tr}
\left(\Lambda_{k+1}^\dagger\frac{\partial U_k}{\partial u_{j,k}}X_k\right).
$$

The Fréchet derivative of the matrix exponential supplies
$\partial U_k/\partial u_{j,k}$ exactly for the discretized dynamics. One forward and
one backward sweep therefore evaluate all slice gradients
<d-cite key="khaneja2005grape"></d-cite>.

The word “exact” refers to the piecewise-constant discretization, not to continuous
hardware. If $A_k=-iH(u_k)\Delta t$ and
$E_{j,k}=-i(\partial H/\partial u_j)\Delta t$, then

$$
\begin{aligned}
D\exp(A_k)[E_{j,k}]
&=\int_0^1 e^{(1-s)A_k}E_{j,k}e^{sA_k}\,ds,\\
\exp\!\begin{pmatrix}A_k&E_{j,k}\\0&A_k\end{pmatrix}
&=\begin{pmatrix}e^{A_k}&D\exp(A_k)[E_{j,k}]\\0&e^{A_k}\end{pmatrix}.
\end{aligned}
$$

The block-exponential identity provides a stable Fréchet
derivative instead of replacing it with the first-order approximation
$-i\Delta t(\partial H/\partial u_j)U_k$. The latter is acceptable only when relevant
commutators are negligible. At finite blockade, $V\Delta t$ is one of the scales that
decides whether that approximation is trustworthy.

Forward products $X_k$ and backward products $\Lambda_k$ turn a naively quadratic
repropagation cost into two sweeps plus local contractions. The terminal phase $\theta$
is reoptimized for the CZ coset; at a smooth maximizer, Danskin's theorem allows the
control derivative to hold the selected $\theta$ fixed. A finite-difference spot check
is still essential because a conjugation or product-order error can produce a plausible
but wrong gradient.

### One slice, worked through

Take three propagators, $U=U_2U_1U_0$, and vary only the first quadrature on the middle
slice. The product rule gives

$$
\frac{\partial U}{\partial\Omega_{x,1}}
=U_2\,D\exp(A_1)[E_{x,1}]\,U_0.
$$

Everything before the varied slice is the forward object $X_1=U_0$; everything after it
is absorbed into the backward object constructed from $U_2$ and the terminal derivative.
For 100 slices the pattern is identical. Each derivative inserts one local Fréchet block
between a cached prefix and suffix, so the cost grows roughly linearly with the number of
slices rather than requiring a complete new rollout for every parameter.

The gradient is only as good as its convention. If time products are ordered from right
to left, the cached prefix must use that same order. If the complex drive is defined as
$\Omega_x+i\Omega_y$, the $y$ generator must carry the matching signs. If controls are
stored in rad/µs but plotted in MHz, the optimizer derivative is taken with respect to the
stored angular quantity, not the displayed value. A robust implementation compares a few
random directional derivatives against centered finite differences before launching a
long search.

Once the gradient is available, GRAPE does not prescribe one universal outer optimizer.
Plain ascent, conjugate gradients, quasi-Newton updates, or constrained optimizers can all
use it. Step selection and initialization influence which local optimum is reached. The
name therefore identifies an efficient derivative construction more than a single
complete optimization algorithm.

The GRAPE search found a $1\,\mu\mathrm s$ pulse whose coherent trace infidelity was
below the saved artifact's numerical reporting threshold. That is an optimizer success,
not yet a physical-gate claim: the pulse has sharp changes, uses no slew constraint, and
has not been priced for decoherence.

## Krotov

Krotov's method also propagates states forward and adjoint states backward, but it updates
the field sequentially while constructing the next trajectory. Under the method's
functional assumptions and update rule, each iteration can be made monotonic
<d-cite key="reich2012krotov"></d-cite>. That property is appealing when an optimizer's
occasional bad step is costly.

The qualifier matters. This implementation enforces the amplitude box by clipping. Clipping is an external
projection, not the unconstrained Krotov update for which the monotonic proof was derived.
The recorded $1\,\mu\mathrm s$ pulse reaches $1.65\times10^{-3}$ infidelity, but its
many saturated slices also say that the update wants to leave the admissible set.

This is a recurring control lesson: an algorithmic theorem applies to a mathematical
formulation, not automatically to every practical modification of its implementation.

In its usual first-order form, the Krotov field update has the schematic structure

$$
u^{(i+1)}(t)=u^{(i)}(t)
+\frac{S(t)}{\lambda_a}
\operatorname{Im}\!\left\langle
\chi^{(i)}(t)\middle|
\frac{\partial H}{\partial u}
\middle|\psi^{(i+1)}(t)
\right\rangle .
$$

The old costate is paired with the newly propagated state. Monotonicity depends on the
functional's convexity or suitable second-order terms, the sign and magnitude of the
step parameter $\lambda_a$, the shape function $S(t)$, consistent propagation, and the
absence of an incompatible post-update operation. Hard clipping can improve feasibility
while breaking the proof because the clipped field is no longer the stationary update
of the stated functional. Constrained variants exist; the point is that the practical
rule must match the theorem being invoked.

### What “sequential” means

A Krotov iteration first propagates the current control forward and uses the terminal
objective to initialize costates, which are stored while moving backward. It then starts
a new forward propagation. At each time, the new state already reflects the controls
updated at earlier times in the same iteration, while the costate comes from the previous
iteration. The new control value is computed from that mixed pair and used immediately.
GRAPE, by contrast, normally forms the gradient of one complete old trajectory and then
updates all slices together.

This temporal ordering is part of the monotonic construction. So are the running cost
that penalizes changes from the reference field and the envelope $S(t)$ that suppresses
updates near endpoints. If the Hamiltonian depends nonlinearly on the field or the
terminal functional is not in the assumed class, second-order terms may be required.
Monotonicity also says only that the chosen functional does not worsen from one iteration
to the next; it does not promise convergence to the global optimum or to a pulse that
satisfies an unstated bandwidth limit.

The saturated Krotov trace on this page can therefore be read diagnostically. Long runs
at a box edge say that the unconstrained update has more authority than the allowed
field. Alternating signs say that independent time samples can still express rapid
changes. Neither pattern should be hidden by smoothing before analysis. A hardware-aware
rerun would need a constraint-compatible update, a filtered parameterization, or a model
that includes the transfer function during optimization.

## CRAB

CRAB replaces $N$ independent slice values by a chopped basis,

$$
u(t)=u_0(t)+S(t)\sum_{m=1}^{M}
\left[a_m\sin(\omega_m t)+b_m\cos(\omega_m t)\right],
$$

where $u_0$ is a seed, $S(t)$ can impose endpoint behavior, and the frequencies are
usually randomized around a useful scale. A derivative-free method then searches the
small coefficient vector <d-cite key="caneva2011crab"></d-cite>.

The reduced dimension can be a feature rather than a compromise. It regularizes the
pulse and can encode bandwidth before optimization begins. It can also be a cage: if the
needed correction is not represented by the selected modes, no amount of coefficient
search will find it.

dCRAB reduces that particular risk by changing the randomized basis after a search
stagnates. Each “super-iteration” opens new directions while retaining the current pulse
as a seed. It can escape an artificial false trap created by one finite basis, but it
does not turn the method into a dense time-grid search: the spectrum, envelope, and
number of modes still define what can be expressed at each stage.

### A bandwidth example

Suppose a 700 ns pulse is expanded in four sinusoidal frequencies near integer multiples
of $1/T$. The coefficient vector may have only eight real entries, yet each coefficient
changes the control over the entire gate. Low modes bend the broad envelope; higher modes
add smaller-scale corrections. Choosing $S(t)=\sin^2(\pi t/T)$ automatically turns those
corrections off at both endpoints, a behavior that would otherwise require extra
constraints on a slice grid.

That compactness also exposes the main risk. If finite blockade requires a correction on
a time scale shorter than the highest chosen period, the basis cannot express it. A
derivative-free optimizer may report convergence because it has exhausted its coefficient
space, even though a better waveform exists outside that space. Randomizing the
frequencies reduces systematic alignment with a poor harmonic grid; changing them in a
dCRAB super-iteration supplies new directions after stagnation.

CRAB is sometimes described as “bandwidth limited,” but the precise statement depends on
the basis and on how the final waveform is synthesized. A finite trigonometric expansion
has bounded spectral support in the mathematical model. Endpoint envelopes multiply that
expansion and broaden its spectrum, while discretization and device interpolation can
add further content. The honest workflow is therefore the same as for a grid pulse:
construct the emitted or filtered waveform and rescore that waveform in the dynamical
model.

With only eight parameters dressing the analytic pulse, the CRAB search reaches
$2.74\times10^{-4}$ infidelity at 683.2 ns. That is worse than the coherent GRAPE
number but more than sixty times better than the undressed finite-blockade baseline, at
the shorter analytic duration.

## Reading the pulses

{% include figure.liquid path="assets/img/neutral-atom-control/part2-method-pulses.png" alt="Three stacked pulse plots comparing GRAPE, Krotov, and CRAB drive quadratures and detuning" caption="Blue is $\Omega_x/2\pi$ and orange is $\Omega_y/2\pi$, the two drive quadratures; green is the detuning $\Delta/2\pi$. All three are ordinary frequencies in MHz. GRAPE and Krotov have 100 free slices and fixed zero detuning—the green zero line is still the plotted detuning control—whereas CRAB uses eight basis parameters around the analytic seed and retains its nonzero detuning." %}

The pulse plot is as informative as the scalar objective. GRAPE uses its freedom to build
a structured but high-bandwidth pattern. Krotov spends many slices on the component
limits. CRAB remains smooth within each analytic segment and preserves the central phase
jump. The algorithms have not discovered three approximations to one canonical waveform;
they have found three different routes through the same unitary dynamics.

The phase representation also matters. Plotting only
$\lvert\Omega\rvert=\sqrt{\Omega_x^2+\Omega_y^2}$ would hide quadrature sign and phase jumps.
For that reason every pulse figure in this series shows $\Omega_x$, $\Omega_y$, and
$\Delta$, or explicitly states when the phase is fixed.

### How to read the three panels

Begin with color rather than method name. Blue and orange are the two Cartesian
components of one complex Rabi drive. Where one component crosses zero, the drive has not
necessarily turned off: the other component may still carry nearly the full amplitude.
Green is a different control, the detuning. In the GRAPE and Krotov panels it lies at
zero by construction; that flat line records a fixed choice, not missing data. In the
CRAB panel it retains the nonzero value inherited from the analytic seed.

Next compare the radius $\sqrt{\Omega_x^2+\Omega_y^2}$ with the component boxes. GRAPE
and Krotov were bounded separately in each quadrature. A long flat segment in one color
therefore indicates saturation of that component, while simultaneous saturation would
correspond to the edge of the larger square allowed in the quadrature plane. CRAB instead
moves along a low-dimensional curve determined by its basis coefficients and seed.

Then look at time scale. A sign reversal between adjacent 10 ns slices is a high-frequency
request even if both values satisfy the amplitude bound. The optimizer was charged no
cost for such a reversal, so rapid changes are free mathematical resources. The plot
connects slice centers to make the sequence visible; it should not be interpreted as a
claim that a laser emits those straight segments. The actual zero-order hold,
interpolation, and analog response belong to the hardware model in Part 5.

Finally compare shape with score. The GRAPE trace has the best recorded closed-system
objective and substantial fine structure. Krotov is even rougher yet ends at a worse
local solution under the clipped update. CRAB is smoother and shorter but retains a
larger coherent error. No single ordering follows until one decides whether the scarce
resource is coherent infidelity, gate time, bandwidth, robustness, or Rydberg exposure.
That is why the later noise comparison can reverse the apparent ranking.

The jaggedness is not a plotting artifact and has not been smoothed in this draft. A
simple, reproducible diagnostic is the quadrature-plane total variation

$$
\mathcal R=
\frac{1}{\Omega_{\max}}
\sum_{k=0}^{N-2}
\left(|\Omega_{x,k+1}-\Omega_{x,k}|
+|\Omega_{y,k+1}-\Omega_{y,k}|\right).
$$

For the saved arrays, $\mathcal R=26.41$ for GRAPE, $73.49$ for Krotov,
$3.64$ for CRAB, and $2.41$ for the analytic two-segment representation. GRAPE and
Krotov were each given independent slice variables, amplitude boxes, and no slew,
curvature, or bandwidth constraint. Their irregularity is therefore an honest feature
of the optimization problem. A later transfer function may smooth it, but that would be
a different emitted pulse and must be rescored.

## The Jandura–Pupillo reference

Jandura and Pupillo ask how quickly globally driven neutral atoms can realize two- and
three-qubit entangling gates when Rydberg blockade is ideal or very strong. Rather than
choosing a convenient duration and minimizing error, they treat duration itself as the
quantity to be minimized. Their numerical quantum-speed-limit analysis identifies short
gate families and shows that a two-qubit CZ can be expressed with the drive held at its
maximum amplitude while only its optical phase varies
<d-cite key="jandura2022time"></d-cite>. This is a particularly useful counterexample to
the idea that a fast gate must look jagged on a time grid.

The published waveform data <d-cite key="jpfigshare"></d-cite> give
$T\Omega_{\max}=7.612$, constant $\lvert\Omega\rvert=\Omega_{\max}$, and a continuously
varying phase. Propagating those tabulated samples in the corresponding
ideal/strong-blockade model gives $1-F_{\mathrm{tr}}=6.43\times10^{-10}$; under the same
roughness definition used above, $\mathcal R=4.45$. These numbers verify the plotted
reference waveform, but they do not turn it into a like-for-like benchmark against the
finite-interaction searches on this page.

{% include figure.liquid path="assets/img/neutral-atom-control/part2-reference-amplitude-phase.png" alt="Amplitude and unwrapped phase comparison for saved GRAPE, CRAB, analytic Levine-Pichler, and published Jandura-Pupillo CZ pulses" caption="All four arrays are plotted without smoothing. GRAPE and CRAB use the finite-blockade two-qutrit model. The published Jandura–Pupillo pulse uses ideal or very strong blockade, normalized phase-only control, and a different duration. It is structural evidence that smooth phase modulation can reach the ideal-blockade limit, not a finite-blockade benchmark against the finite-interaction searches." %}

This comparison is deliberately asymmetric. Mapping
$\Omega_{\max}=2\pi\times2$ MHz turns the published normalized duration into about
606 ns, but its Hamiltonian limit and parameterization differ from the finite-blockade
comparison. It would be
misleading to place its infidelity beside the finite-blockade rows and declare a solver
winner. What it does establish is more useful: a high-fidelity time-optimal solution can
live in a smooth, constant-amplitude, phase-only language. The jagged grid-optimized pulses are a
consequence of those saved formulations, not a fundamental requirement of fast CZ
control.

Allowing detuning to vary adds a control direction. In the duration sweep, the
three-control search moves the numerical frontier toward shorter gates. Close to a speed
limit, one more control is not a small convenience: it can change which paths through
Hilbert space are reachable at a given time.

{% include figure.liquid path="assets/img/neutral-atom-control/part2-duration-frontier.png" alt="Log-scale CZ infidelity versus duration for searches with two and three controls, with an ideal-blockade speed-limit reference" caption="The dashed 605.7 ns line is the ideal-blockade time-optimal reference, not a finite-blockade hardware guarantee. Releasing $\Delta(t)$ generally moves the sampled frontier left." %}

Entries saved as floating-point zero are drawn at $10^{-13}$ with a downward marker and
read as $1-F_{\mathrm{tr}}\le10^{-13}$. Literal zero has no location on a logarithmic
axis and would overstate numerical knowledge. The lightweight duration check in the
separate reproduction code is intentionally excluded: it was never intended to replace
the paper's speed-limit analysis.

Convergence curves also need interpretation. A flat objective may mean that the current
parameterization has reached a local optimum, that the gradient is poorly scaled, that a
box constraint blocks the useful direction, or simply that numerical precision has been
reached. Restarting from several seeds probes the first possibility; checking directional
derivatives probes the second; inspecting active bounds probes the third. None of these
tests alone certifies a global optimum.

A stopping tolerance should be tied to the next physical error scale. Continuing a
closed-system search from $10^{-8}$ to $10^{-12}$ is difficult to justify if expected
dissipative error is $10^{-2}$ and the waveform becomes rougher in the process. Conversely,
a loose optimizer tolerance can obscure a speed-limit study whose purpose is to resolve a
narrow duration frontier. Numerical precision is a resource chosen for the scientific
question, not a universal badge of pulse quality.

## A fair comparison

The recorded outcomes are best read as demonstrations of strategy:

| Pulse             | Duration | Coherent CZ trace infidelity | Structural caveat                    |
| ----------------- | -------: | ---------------------------: | ------------------------------------ |
| analytic baseline | 683.2 ns |          $1.80\times10^{-2}$ | finite-blockade error                |
| CRAB              | 683.2 ns |          $2.74\times10^{-4}$ | eight-parameter seeded ansatz        |
| Krotov            |  1000 ns |          $1.65\times10^{-3}$ | clipped component boxes              |
| GRAPE             |  1000 ns |        below saved precision | unconstrained bandwidth and exposure |

Five questions should be answered before reading any two rows as a competition.

First, do they use the same Hamiltonian? The three finite-blockade rows do, whereas the
Jandura–Pupillo reference deliberately uses an ideal/strong-blockade reduction. Second,
do they optimize the same duration? A longer pulse has more time to accumulate a phase
but also more time to decohere. Third, do they expose the same controls? Releasing
$\Delta(t)$ can alter reachability near a speed limit. Fourth, do they enforce the same
feasible set? Component boxes, a circular amplitude bound, a smooth basis, and explicit
slew constraints describe different sets of waveforms. Fifth, are scores computed with
the same target phase and leakage convention?

Initialization deserves a sixth question in practice. The CRAB run begins near an
analytic solution; a randomly initialized dense-grid method begins elsewhere in the
landscape. Multiple starts can estimate sensitivity to that choice, but they do not prove
global optimality. Wall-clock time is also implementation dependent: matrix size,
derivative code, tolerances, parallelism, and language overhead can dominate the abstract
method. For this reason the table reports scientific outcomes and structural caveats,
not timings presented as a package benchmark.

They do not share duration, initialization, detuning freedom, or constraint treatment.
Declaring a universal winner would confuse solver performance with problem definition.
The robust conclusion is narrower: flexible gradients can nearly erase coherent error;
a compact basis can make a strong short pulse; and neither approach natively expresses
the full set of hardware inequalities we will want next.

## The method landscape

GRAPE, Krotov and CRAB are the three routes this chapter compares directly, but each is the head of a family, and the families differ in ways the comparison above does not surface.

<table class="nac-metric-table">
<thead><tr><th>Method</th><th>Search space</th><th>Gradient</th><th>Monotonic?</th><th>Open-system native?</th></tr></thead>
<tbody>
<tr><td>GRAPE <d-cite key="khaneja2005grape"></d-cite></td><td>Full time grid</td><td>Analytic, concurrent update</td><td>No</td><td>Via extension <d-cite key="schulteherbruggen2011opengrape"></d-cite></td></tr>
<tr><td>Krotov <d-cite key="palao2003unitary,reich2012krotov"></d-cite></td><td>Full time grid</td><td>Analytic, sequential update</td><td>Yes, under stated conditions</td><td>Yes <d-cite key="goerz2014krotovopen"></d-cite></td></tr>
<tr><td>CRAB <d-cite key="caneva2011crab"></d-cite></td><td>Low-dimensional random basis</td><td>Gradient-free</td><td>No</td><td>Indirectly</td></tr>
<tr><td>dCRAB <d-cite key="rach2015dcrab"></d-cite></td><td>Re-randomized basis, iterated</td><td>Gradient-free</td><td>No</td><td>Indirectly</td></tr>
<tr><td>GOAT <d-cite key="machnes2018goat"></d-cite></td><td>Explicit parametric pulse</td><td>Analytic, via coupled ODEs</td><td>No</td><td>Extensible</td></tr>
</tbody>
</table>

Read as history: GRAPE <d-cite key="khaneja2005grape"></d-cite> and Krotov <d-cite key="palao2003unitary,reich2012krotov"></d-cite> both search the full time-discretized control but differ in whether amplitudes are updated concurrently or sequentially — and that difference is what buys Krotov its monotonicity guarantee. Machnes and coauthors put both into a single framework and benchmarked them against one another, which is the reference this chapter's own comparison is implicitly echoing <d-cite key="machnes2011comparing"></d-cite>. CRAB moved in the opposite direction, restricting the search to a handful of random basis coefficients so that the optimization could run on hardware where no gradient is available <d-cite key="caneva2011crab"></d-cite>; dCRAB then re-randomizes that basis between rounds to escape the artificial traps a fixed truncation introduces <d-cite key="rach2015dcrab"></d-cite>. GOAT returned to gradient-based search while keeping the pulse in an explicitly parametric, hardware-realizable form <d-cite key="machnes2018goat"></d-cite>.

<div class="nac-callout">
<strong>On Krotov's monotonicity.</strong> The guarantee is real but conditional. It relies on a quadratic-in-the-control cost and a Hamiltonian linear in the control; when the functional is higher-order in the states, the dynamics are non-unitary, or the control enters nonlinearly, a second-order term is needed to preserve it <d-cite key="reich2012krotov"></d-cite>. Quoting "Krotov converges monotonically" without those conditions is the most common overstatement about this family.
</div>

Two broader reviews cover this landscape in far more depth than a comparison table can, and are the right entry points for a reader who wants the full picture <d-cite key="glaser2015cat,koch2022review"></d-cite>. For reproducing Krotov specifically, a maintained implementation exists <d-cite key="goerz2019krotovpackage"></d-cite>.

## Laboratory note

<div class="nac-lab-note"><strong>What changed.</strong> Closed-system fidelity stopped being the only interesting axis. The GRAPE pulse made the model-level CZ essentially exact, but its sharp quadratures and long Rydberg exposure created liabilities that were invisible to the objective. CRAB's less spectacular scalar score came with a shorter, smoother waveform.</div>

The next step is not simply “run GRAPE longer.” It is to ask for endpoint, amplitude,
slew, curvature, fidelity, and duration requirements _simultaneously_. Direct collocation
makes that request explicit.

{% include neutral_atom_control/series_nav.liquid %}
