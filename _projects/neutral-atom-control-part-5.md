---
layout: distill
title: "5 · From Optimized Arrays to Hardware Candidates"
description: Pulser sequences, clocks, channel modulation, geometry, anonymized SPAM calibration, and an unexecuted run plan
permalink: /projects/neutral-atom-control/part-5-hardware-bridge/
tags: quantum-control neutral-atoms Pulser hardware
giscus_comments: false
importance: 99
category: work
show_on_projects: false
series: neutral-atom-control
series_part: 5
series_previous_url: /projects/neutral-atom-control/part-4-noise-robustness/
series_previous_label: "Part 4"
status: draft
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: neutral-atom-control.bib
toc:
  - name: A deliberate change of task
  - name: Arrays are not waveforms
  - name: What the channel delivers
  - name: Geometry is a constraint
  - name: Aggregate hardware calibration
  - name: The proposed run
  - name: What remains for native CZ
---

{% include neutral_atom_control/series_nav.liquid %}

## A deliberate change of task

Parts 1–4 optimize a gate in an effective three-level basis
$\{\lvert0\rangle,\lvert1\rangle,\lvert r\rangle\}$ for each atom. The available Pulser
<code>AnalogDevice</code> abstraction exposes a globally driven two-level
$\{\lvert g\rangle,\lvert r\rangle\}$ manifold. It does not, by itself, encode the same
computational qubit, local phase corrections, and three-level CZ semantics.

Pretending otherwise would make the cleanest-looking result in the series scientifically
wrong. The hardware bridge therefore uses a smaller hardware-native problem: prepare

$$|\Psi^+\rangle=\frac{|gr\rangle+|rg\rangle}{\sqrt2}$$

from $\lvert gg\rangle$, using global $\Omega(t)\ge0$, detuning $\Delta(t)$, and the same
$C_6/R^6$ interaction. Bell preparation exercises blockade, entanglement, waveform
translation, noise, and measurement without claiming that the earlier CZ was executed.

This change of task is a model-interface decision, not a statement that Bell preparation
is equivalent to a two-qubit gate. A state-transfer experiment tests one input and one
target ray. A gate must act correctly on every superposition in the computational
subspace. In particular, the Bell target is insensitive to several phases that a CZ
objective must retain. High Bell fidelity can validate blockade dynamics and the delivery
pipeline while leaving the native-CZ question open.

The distinction also prevents a basis-label shortcut. Relabeling $\lvert g\rangle$ as
$\lvert1\rangle$ does not create the dark $\lvert0\rangle$ level or the local phase
corrections assumed by the qutrit model. A hardware-facing demonstration must either use
a device abstraction that represents those levels and couplings or specify an encoded
protocol that realizes the same logical map. Until then, the honest claim is a
hardware-feasible blockade-mediated entangled-state candidate.

{% include figure.liquid path="assets/img/neutral-atom-control/hardware-pipeline.png" alt="Pipeline from optimized arrays through a Pulser sequence, clock and device constraints, channel modulation, and delivered rollout" caption="A hardware bridge is a chain of model transformations. Every arrow can change the answer." %}

## Arrays are not waveforms

The two Bell-state candidates begin as arrays in rad/µs:

| Spacing | Optimizer duration | Slices | Coherent Bell infidelity |
| ------: | -----------------: | -----: | -----------------------: |
|  5.0 µm |             300 ns |     30 |          below $10^{-8}$ |
|  6.5 µm |            1500 ns |     75 |          below $10^{-8}$ |

The exporter converts each into a Pulser <code>Sequence</code>
<d-cite key="silverio2022pulser"></d-cite>. Two transformations are already lossy:

1. Duration must lie on the channel's 4 ns clock. The exporter rounds down rather than
   silently exceeding the device duration after fall time is added.
2. Slice-center values become an interpolated waveform. The export path uses shape-preserving
   interpolation rather than treating the optimizer's array as an analog signal.

If the channel clock is $\tau_c=4$ ns, a requested duration becomes an integer
$N_c\tau_c$. The rounding policy is part of the experiment: rounding upward can violate
a maximum duration once the fall time is included, whereas rounding downward changes
the integrated pulse area. The audit records both the optimizer grid and serialized clock grid
so this change is inspectable.

### A clock-rounding example

Suppose an optimizer returns a 350 ns pulse while the channel accepts durations only in
4 ns increments. The neighboring clock-compatible durations are 348 ns and 352 ns. If an
85 ns fall-time allowance must also fit beneath a hard sequence limit, choosing 352 ns
may make the complete request invalid even though it is closer to the optimizer value.
Choosing 348 ns avoids that overrun but removes two nanoseconds of pulse area and shifts
every normalized sample time.

That two-nanosecond change is small compared with the gate, yet it is not mathematically
zero. For a constant angular Rabi rate $\Omega$, shortening by $\delta t$ changes rotation
area by $\Omega\delta t$. For a shaped waveform it also changes interpolation locations
and the phase accumulated under detuning. The exported sequence must therefore be scored
after quantization; recording only the nominal optimizer fidelity would skip the first
hardware transformation.

The time-grid convention matters just as much. Thirty optimizer values over 300 ns can
mean thirty 10 ns holds, thirty slice-center samples, or thirty knots including endpoints.
Those interpretations have different supports. The exporter treats the stored values
according to their recorded convention, constructs the requested continuous curve, and
then samples it on the device grid. A provenance record should preserve all three axes:
optimizer time, programmed clock time, and delivered-channel time.

Interpolation answers a separate question. An optimizer value may represent a slice
center, a left endpoint, or a knot of a continuous interpolant. Treating those
conventions as interchangeable creates a half-step timing error. Shape-preserving cubic
interpolation avoids some overshoot, but it does not manufacture bandwidth feasibility;
it only defines the requested continuous curve sampled by Pulser.

The sequence then checks amplitude, detuning, clock alignment, minimum and maximum
duration, atom spacing, radial extent, and delivered slew. It is serialized to Pulser's
abstract representation and reconstructed; the nominal samples are required to survive
that round trip exactly.

That serialization round trip checks units, duration, waveform support, register
coordinates, and phase segmentation. It does not check the physics of the emitted
optical field. The abstract representation is a device-facing program, not a certificate
that the requested array is reachable after the analog channel.

The analytic Levine–Pichler CZ supplies a useful representation test. Its one phase jump
can be represented by two consecutive constant-phase pulses. A continuously
phase-modulated CZ pulse would require many such segments. Exact syntax for a phase
boundary does not guarantee accurate device physics between boundaries.

A complex control can be written $\Omega_x+i\Omega_y=Ae^{i\phi}$. Pulser's global
channel expresses nonnegative amplitude plus phase. A piecewise-constant phase jump can
therefore be segmented exactly at a clock boundary. A rapidly winding phase would need
many segments or a phase-modulation primitive, and finite phase-update bandwidth would
become another transfer function. This is why the quadrature plots in Part 2 cannot be
translated by plotting $\lvert\Omega\rvert$ alone.

Phase conversion has a delicate corner at vanishing amplitude. The angle
$\phi=\operatorname{atan2}(\Omega_y,\Omega_x)$ is undefined when both quadratures are
zero, even though the physical drive is off. An exporter must carry the previous phase,
choose a documented default, or omit a zero-amplitude segment; arbitrary phase jumps at
zero can otherwise create needless instructions. Phase unwrapping is only a display and
segmentation aid because phases differing by $2\pi$ describe the same complex field.

Segmentation also introduces a resolution choice. Approximating a continuously varying
phase with more constant-phase pieces reduces representation error but increases program
complexity and may exceed update-rate constraints. Fewer pieces are easier to serialize
but can alter the trajectory. The correct number is not determined by how smooth a plot
looks; it is established by convergence of the delivered-waveform rollout under finer
segmentation and by the actual phase-control specification.

## What the channel delivers

The programmed sequence is not the emitted optical field. The channel has an 8 MHz
modulation bandwidth and an 85 ns rise time. Pulser's modulation model adds rise and fall
tails and smooths fast structure. The analysis samples both the nominal and modulated sequences,
then independently propagates the delivered arrays.

The delivered grid is longer than the programmed grid because the filter rings up and
down. In the figure, the blue curve ends at the programmed duration while the orange
curve continues through the shaded tail on its own time axis. Comparing both arrays
against optimizer slice index would hide this extra evolution.

{% include figure.liquid path="assets/img/neutral-atom-control/part5-programmed-delivered.png" alt="Four panels comparing programmed and Pulser-modulated Rabi and detuning waveforms at 5.0 and 6.5 micrometer spacing" caption="The curves are exact samples from Pulser with modulation disabled and enabled, not an illustrative filter. Controls are shown as $\Omega/2\pi$ and $\Delta/2\pi$ in MHz; the horizontal coordinate is the 1 ns sampled hardware clock." %}

The shaded tail is dynamically active. Once the programmed samples stop, the filtered
field does not instantly become zero. The state continues to evolve under the residual
drive, detuning, and interaction until the modeled output settles. Truncating propagation
at the nominal optimizer duration would therefore score a field that the channel never
delivers. Conversely, appending zeros to the optimizer array without filtering would not
reproduce the same ring-down.

Bandwidth changes phase as well as amplitude. Filtering $\Omega_x$ and $\Omega_y$
separately can alter the instantaneous angle of the complex envelope. A model expressed
as nonnegative amplitude and phase may implement a different internal transfer path. The
export audit should follow the representation used by the device abstraction and compare
the reconstructed complex field, not assume that smoothing the amplitude alone captures
the channel.

The modest fidelity reduction for these Bell candidates is an observed property of this
specific translation. It should not be read backward as proof that the irregular samples
were physically smooth, or forward as a guarantee for other pulses. A resonance between
filter delay and a phase jump could produce a much larger error. Delivered-waveform
propagation is therefore a required scoring stage for every candidate, not a one-time
validation of the export code.

At $R=5\,\mu\mathrm m$, the sequence representation gives
$F=0.999980$ before modulation and $F=0.999915$ after it. At
$R=6.5\,\mu\mathrm m$, the corresponding values are $0.999967$ and $0.999058$.
The programmed Bell arrays are visibly irregular. They nevertheless retain high
simulated Bell fidelity after this particular Pulser modulation model. That is an
empirical result for these two saved arrays, this state-transfer target, and this filter;
it is not “smoothness by construction,” and it cannot be generalized to the jagged CZ
pulses from Part 2 without translating and independently propagating those pulses too.

This closes a loop back to Part 3. Slew limits did not merely make the collocation plots
look nicer. They were a way of buying robustness to the physical channel before the
Pulser representation existed.

## Geometry is a constraint

The abstract sequence can place two atoms at any validated continuous coordinates. A
real machine can expose a calibrated trap layout with only certain pair separations. In
the selected device configuration, 5.0 µm is available but 6.5 µm is not; the next useful
spacing is 8.66 µm.

That means the 6.5 µm pulse is a simulator candidate, not a machine-ready candidate. It
should be reoptimized at a realizable spacing rather than rounded to the nearest trap.
Because $V=C_6/R^6$, a geometric change is a Hamiltonian change, not a minor placement
error.

{% include figure.liquid path="assets/img/neutral-atom-control/part5-geometry-blockade.png" alt="Interaction strength and dimensionless blockade ratio at atom spacings 5.0, 6.5, and 8.66 micrometers" caption="The $R^{-6}$ law makes geometry a control parameter. At 8.66 µm the interaction and blockade ratio are far below their 5.0 µm values, so moving a pulse to the nearest available pair is not a harmless coordinate edit." %}

The three points use the same recorded $C_6$, not a fitted curve. The 6.5 µm point is
valuable for simulation but absent from the selected calibrated layout. The 8.66 µm
point is available, yet its much weaker blockade defines a different control problem. A
sound workflow enumerates feasible geometry first and optimizes on each candidate
Hamiltonian.

Geometry also connects pulse-level control to analog neutral-atom programming. In an
analog independent-set protocol, atom coordinates encode graph adjacency through the
blockade radius; the [unit-disk mapping tutorial](/blog/2025/UnitDiskMapping/) develops
that workflow in detail. Here only one pair is considered, but the design rule is the
same: coordinates enter the Hamiltonian before they enter a drawing. A layout validator
cannot repair a pulse optimized for the wrong $V$.

At 8.66 µm, the interaction is reduced by the sixth power of the spacing ratio. The
single-excitation drive can then compete with, rather than be dominated by, the shift of
$\lvert rr\rangle$. Population paths and conditional phase change together. Retuning
duration alone may be insufficient; amplitude, detuning, and waveform shape should all be
optimized for the available pair. If several calibrated pairs exist, geometry can be an
outer discrete choice and pulse design an inner continuous optimization.

Real layouts add uncertainty around nominal coordinates. Thermal motion and trap
calibration produce a distribution of $R$, which becomes a strongly nonlinear
distribution of $V$. A hardware candidate can include several interaction strengths in
its robust ensemble, but the range should come from position calibration. This is another
reason to postpone the label “hardware pulse” until register choice and its uncertainty
are part of the model.

## Aggregate hardware calibration

The calibration model uses aggregate counts from two prior QPU batches totaling 1,100 shots. The public
draft intentionally removes job IDs, account/team metadata, layout identifiers, and the
submission recipe. A simple readout-and-preparation model fits:

| Parameter               | Fitted probability |
| ----------------------- | -----------------: |
| false positive          |             1.388% |
| false negative          |             8.179% |
| state-preparation error |             0.326% |

The asymmetry is important. A single symmetric “measurement error” would hide the large
false-negative component. Within this simple model, the observed floor is dominated by
state preparation and measurement (SPAM); the residual assigned to coherent dynamics is
less than roughly two percentage points. This is an inference from a small calibration
set, not a complete device-noise reconstruction.

For one atom, a minimal asymmetric readout model is

$$
\begin{pmatrix}P(m=0)\\P(m=1)\end{pmatrix}
=
\begin{pmatrix}
1-p_{\mathrm{fp}} & p_{\mathrm{fn}}\\
p_{\mathrm{fp}} & 1-p_{\mathrm{fn}}
\end{pmatrix}
\begin{pmatrix}P(0)\\P(1)\end{pmatrix}.
$$

The false-positive and false-negative probabilities need not match. State-preparation
error is applied before this confusion matrix, so it should not be folded into one
“readout fidelity.” With aggregate counts, the fitted decomposition is useful for
planning but not uniquely diagnostic of every physical mechanism.

For two atoms, the simplest extension takes a tensor product of the one-atom confusion
matrices. That assumes independent readout errors. Crosstalk, correlated loss, or a
state-dependent detection process would require additional parameters that aggregate
counts may not identify. State-preparation error is likewise not uniquely separated from
early dynamical error without dedicated calibration sequences. The fitted values should
therefore be used to forecast observed populations, not narrated as a complete microscopic
noise model.

Inverting a confusion matrix can estimate pre-readout populations, but mitigation trades
bias for variance. A large false-negative asymmetry can amplify shot noise, and an
unconstrained inverse may even produce negative estimated probabilities in a finite
sample. A validation report should show raw counts first, document the calibration matrix
and its uncertainty, and present mitigated values as a secondary analysis with propagated
confidence intervals.

{% include figure.liquid path="assets/img/neutral-atom-control/part5-spam-run-plan.png" alt="Bar plots of fitted false-positive, false-negative, and preparation probabilities, and ideal versus calibrated predicted Bell population" caption="Only aggregate calibration enters the draft. The predicted $P_{\mathrm{Bell}}=0.898$ belongs to the calibrated model; it is not a measured result for the proposed pulse." %}

The roughly ten-percent gap between an ideal Bell target and a calibrated observed
population is therefore not evidence that another decimal of coherent optimizer fidelity
will help. Better discrimination, measurement mitigation, and a validation protocol
that separates SPAM from dynamics are higher-leverage.

## The proposed run

A robust-GRAPE Bell candidate at $R=5\,\mu\mathrm m$ is selected for the proposed run:

- 350 ns and 35 optimization slices;
- amplitude and detuning exported to a clock-compatible Pulser sequence;
- 1,000 proposed shots;
- calibrated prediction $P_{\mathrm{Bell}}\approx0.898$; and
- a preregistered comparison against aggregate prior behavior.

<div class="nac-callout"><strong>No job was submitted.</strong> The 1,000-shot item is a run plan. The pulse arrays and predicted distribution are simulation artifacts. The only measured information used here is the anonymized aggregate calibration from prior runs.</div>

A useful validation would record, before looking at the outcome, the target statistic,
confidence interval or acceptance band, treatment of invalid/missing atoms, and the
decision rule for whether the new candidate improves on the control. Without that
preregistration, it is too easy to move between parity, Bell population, raw success,
and mitigated success after the counts arrive.

For 1,000 independent shots and a predicted success probability near 0.898, the binomial
standard error is

$$
\sqrt{p(1-p)/1000}\approx0.0096.
$$

A practical protocol should include a contemporaneous reference sequence, predeclare
whether raw or SPAM-corrected counts determine acceptance, and attach a confidence
interval to their difference. It should record atom-loss handling, register validation,
pulse hash, software/device version, and the exact abstract sequence. These safeguards
separate a failed waveform from drift, layout mismatch, or a post-processing choice.

One concrete protocol interleaves candidate and reference batches rather than running all
reference shots first. Interleaving reduces confounding from slow drift. Each batch logs
the validated register and sequence hash, and analysis uses a fixed rule for missing
atoms. The primary endpoint can be the raw probability of the two single-excitation
outcomes that define the Bell-population proxy; a phase-sensitive parity or tomography
measurement would be required to certify coherence rather than population alone.

The acceptance rule should compare the candidate with the contemporaneous reference, not
only with a historical point estimate. For example, predeclare a minimum improvement and
require the lower confidence bound on the difference to exceed it. If SPAM correction is
reported, calibrations collected in the same session should be propagated through the
uncertainty analysis. With 1,000 shots, statistical precision is around one percentage
point, so claims about improvements much smaller than that need either more shots or a
paired design with lower variance.

Four evidence levels remain distinct:

| Level                       | What has been established                                                |
| --------------------------- | ------------------------------------------------------------------------ |
| simulation                  | a mathematical pulse reaches the target in a recorded model              |
| hardware-compatible         | Pulser validates and serializes the selected device abstraction          |
| prior aggregate calibration | anonymized historical counts constrain a simple SPAM model               |
| proposed experiment         | a 1,000-shot candidate and decision protocol exist; no job was submitted |

Moving down this table requires new evidence; it is not a change of wording. Simulation
needs reproducible arrays, units, and a dynamical model. Hardware compatibility adds a
validated device description and serialized sequence. A measured claim adds execution
records, counts, contemporaneous calibration, and an analysis protocol. Passing one level
does not imply the next, and failure at a later level does not erase what was learned at
an earlier one.

This evidence ladder is useful beyond neutral atoms. It prevents a simulator object from
being described as an emitted waveform and prevents a calibrated prediction from being
described as an observation. Keeping hashes and transformations makes disagreements
traceable: one can ask whether the optimizer array, exported sequence, delivered model,
or measurement interpretation first changed the conclusion.

## What remains for native CZ

The hardware bridge does not end the original problem. A native CZ demonstration still
needs:

1. a device model that represents the computational $\lvert0\rangle,\lvert1\rangle$ states and
   their coupling through $\lvert r\rangle$, or a defensible encoded protocol in the available
   interface;
2. explicit handling of local $Z$ phases and the CZ coset;
3. optimization at a realizable trap spacing;
4. channel modulation inside the optimization or robust ensemble;
5. a noise model calibrated from suitable experiments; and
6. a measurement protocol capable of distinguishing a gate from one state-preparation
   trajectory.

<div class="nac-lab-note"><strong>What changed.</strong> The hardware bridge does not force the three-level CZ into a two-level API. It changes to a Bell-state surrogate, carries that pulse through the full sequence and modulation path, and stops at a concrete, anonymized run plan. The boundary between “simulated,” “hardware compatible,” and “measured” remains visible.</div>

That boundary is the main lesson of the series. Optimal-control software is necessary
because a realistic pulse is a constrained trajectory through several models. It is not
sufficient because every model transition—atomic reduction, target gauge, discretization,
noise, channel, geometry, and measurement—must be validated in its own language.

{% include neutral_atom_control/series_nav.liquid %}
