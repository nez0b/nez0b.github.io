---
layout: distill
title: "1 · From Rydberg Atoms to an Optimal-Control Problem"
description: The physical ladder, effective two-qutrit Hamiltonian, blockade, CZ, costates, and PMP
permalink: /projects/neutral-atom-control/part-1-foundations/
tags: quantum-control neutral-atoms rydberg PMP
giscus_comments: false
importance: 99
category: work
show_on_projects: false
series: neutral-atom-control
series_part: 1
series_next_url: /projects/neutral-atom-control/part-2-grape-krotov-crab/
series_next_label: "Part 2"
status: draft
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: neutral-atom-control.bib
toc:
  - name: Two pictures of one atom
  - name: Two atoms and blockade
  - name: What counts as CZ
  - name: Why optimize
  - name: Costates and PMP
  - name: Laboratory note
---

{% include neutral_atom_control/series_nav.liquid %}

## Two pictures of one atom

The laboratory picture begins with an optical ladder. A ground state $\lvert g\rangle$ is
coupled to an intermediate excited state $\lvert e\rangle$, which is coupled again to a
highly excited Rydberg state $\lvert r\rangle$. Two lasers with Rabi frequencies
$\Omega_1$ and $\Omega_2$ drive the two legs. The intermediate state is useful as a
bridge but undesirable as a destination: it is short-lived.

When the one-photon detuning $\lvert\Delta_e\rvert$ is large compared with the laser couplings,
$\lvert e\rangle$ is only virtually occupied. Adiabatic elimination gives an effective
two-photon coupling $\Omega_{\mathrm{eff}}\sim\Omega_1\Omega_2/(2\Delta_e)$, together
with light shifts absorbed into an effective detuning. The numerical gate model starts
_after_ this reduction.

It is worth seeing what has been discarded. In the rotating-wave picture, amplitudes
$c_g,c_e,c_r$ obey a three-level Schrödinger equation whose excited-state row is

$$
i\dot c_e=-\Delta_e c_e+\frac{\Omega_1}{2}c_g
+\frac{\Omega_2^*}{2}c_r.
$$

For slow envelopes and $\lvert\Delta_e\rvert\gg\lvert\Omega_1\rvert,\lvert\Omega_2\rvert$, set
$\dot c_e\simeq0$. Then

$$
c_e\simeq\frac{\Omega_1c_g+\Omega_2^*c_r}{2\Delta_e},
\qquad
\Omega_{\mathrm{eff}}=\frac{\Omega_1\Omega_2}{2\Delta_e}.
$$

Substitution produces both the two-photon coupling and AC Stark shifts
$\lvert\Omega_1\rvert^2/(4\Delta_e)$ and $\lvert\Omega_2\rvert^2/(4\Delta_e)$. Calling the remaining knob
$\Delta(t)$ therefore includes a convention: it is the two-photon detuning after the
chosen light-shift calibration. The minus sign in $-\Delta\lvert r\rangle\langle r\rvert$ below
follows the rotating-frame convention used throughout this series. Reversing that convention is legal,
but only if the control arrays and every comparison are transformed with it.

{% include figure.liquid path="assets/img/neutral-atom-control/levels.png" alt="Energy-level diagram showing a physical ground-intermediate-Rydberg ladder reduced to an effective zero-one-Rydberg control model" caption="The physical ladder motivates the control, while the CZ study optimizes the effective $\lvert0\rangle,\lvert1\rangle,\lvert r\rangle$ model. The intermediate $\lvert e\rangle$ is not a propagated level." %}

The qubit is stored in two long-lived ground or hyperfine states, $\lvert0\rangle$ and
$\lvert1\rangle$. The chosen laser couples $\lvert1\rangle\leftrightarrow\lvert r\rangle$, while
$\lvert0\rangle$ is dark. In the rotating frame and with $\hbar=1$,

$$
H_1(t)=\frac{\Omega_x(t)}{2}(|1\rangle\langle r|+|r\rangle\langle1|)
+\frac{\Omega_y(t)}{2}(-i|1\rangle\langle r|+i|r\rangle\langle1|)
-\Delta(t)|r\rangle\langle r|.
$$

Writing a complex Rabi frequency $\Omega=\Omega_x+i\Omega_y$ makes clear that the two
quadratures control both envelope and phase. Detuning supplies a third control direction.

### When the effective model is trustworthy

Adiabatic elimination is an approximation with a physical operating regime, not a change
of notation. Three time scales must be separated. The optical detuning must be the fastest
scale, the laser envelope must change slowly compared with $1/\lvert\Delta_e\rvert$, and
the desired two-photon evolution must remain slow enough that the virtually populated
intermediate state can follow it. A useful back-of-the-envelope estimate is
$P_e\sim(\lvert\Omega_1\rvert^2+\lvert\Omega_2\rvert^2)/(4\Delta_e^2)$. Even when this
population is small, spontaneous emission from $\lvert e\rangle$ can matter because the
gate samples it for hundreds of nanoseconds.

The reduced model therefore makes a specific promise: it should reproduce the slow
ground–Rydberg dynamics after the light shifts have been calibrated. It does not promise
to predict off-resonant scattering, laser phase noise on each optical leg, or failures of
the rotating-wave approximation. Those effects can be added later as effective decay and
dephasing channels, but they should not be mistaken for phenomena already present in the
three-level qutrit used below.

There is also a bookkeeping issue. The two logical states are called $\lvert0\rangle$ and
$\lvert1\rangle$, whereas the optical derivation began with $\lvert g\rangle$. In this
series, $\lvert1\rangle$ is the ground-state level addressed by the effective Rydberg
transition and $\lvert0\rangle$ is dark. Thus the propagated single-atom basis is
$\{\lvert0\rangle,\lvert1\rangle,\lvert r\rangle\}$, not
$\{\lvert g\rangle,\lvert e\rangle,\lvert r\rangle\}$. Keeping those two pictures
separate prevents a common error: interpreting leakage into $\lvert r\rangle$ as if it
were the eliminated intermediate-state population.

Before optimization, the effective Hamiltonian should pass a few mechanical checks.
Every control generator must be Hermitian, so real control values produce a Hermitian
$H(t)$ and a unitary coherent propagator. Angular frequencies in rad/µs must be used with
time in µs; the plotted division by $2\pi$ is only a display conversion to MHz. Setting
all controls to zero should leave computational basis states unchanged apart from any
explicit drift, and exchanging the two atoms should commute with the global Hamiltonian.
These simple tests catch sign, basis-order, and unit errors before a sophisticated solver
turns them into an apparently excellent pulse.

Time ordering is the next layer. Because $H(t)$ at different times generally does not
commute, the gate is a time-ordered exponential rather than the exponential of an average
Hamiltonian. Piecewise-constant simulation approximates it by an ordered product of
short propagators. Refining the time step should stabilize populations, phases, and the
terminal score. Agreement of only the population curve is insufficient because a small
phase bias can leave the blockade picture looking plausible while moving the gate outside
the CZ coset.

Finally, the effective model defines what the optimizer is allowed to exploit. If the
intermediate state, spatial inhomogeneity, or optical phase response is absent, the
optimizer cannot trade against those effects responsibly. A later hardware model can
rescore the result, but large disagreement then indicates missing physics, not merely a
need for more optimizer iterations. Model reduction and pulse optimization should be
viewed as a loop: reduce, validate in its regime, optimize, and test the candidate in the
next richer model.

## Two atoms and blockade

For two globally driven atoms separated by $R$,

$$
H(t)=H_1(t)\otimes I+I\otimes H_1(t)+V(R)|rr\rangle\langle rr|,
\qquad V(R)=\frac{C_6}{R^6}.
$$

The basis has nine states. The interaction shifts only $\lvert rr\rangle$, but that shift
changes the entire reachable dynamics. At the canonical point used here,
$R=5\,\mu\mathrm m$, $\Omega_{\max}=2\pi\times2$ MHz, and
$V=55.406$ rad/µs. The blockade ratio $V/(\sqrt2\Omega_{\max})\approx3.1$ is
large enough to suppress double excitation, but not infinite.

The global symmetry is valuable. The nine-dimensional evolution separates into a dark
one-dimensional block, two identical two-dimensional blocks, and a symmetric
three-dimensional block. The collocation formulation exploits that structure to reduce each knot to
ten real propagated variables rather than a full $9\times9$ complex unitary.

The blocks can be read directly from the globally driven basis. $\lvert00\rangle$ is dark.
The inputs $\lvert01\rangle$ and $\lvert10\rangle$ each evolve in an identical copy of
$\operatorname{span}\{\lvert01\rangle,\lvert0r\rangle\}$ or
$\operatorname{span}\{\lvert10\rangle,\lvert r0\rangle\}$. Finally, $\lvert11\rangle$ reaches only
the symmetric ladder

$$
|11\rangle
\longleftrightarrow
|W_r\rangle=\frac{|1r\rangle+|r1\rangle}{\sqrt2}
\longleftrightarrow |rr\rangle,
$$

with the first coupling enhanced by $\sqrt2$. The antisymmetric combination is dark.
This is also why the useful blockade ratio contains $\sqrt2\Omega_{\max}$ rather than
$\Omega_{\max}$ alone. Finite $V$ does not delete $\lvert rr\rangle$; it makes it off
resonant. Residual double excitation can return with a phase, leak at the endpoint, or
interfere destructively with another path.

The symmetric ladder gives a concrete way to read blockade. Starting from
$\lvert11\rangle$, the global field couples to $\lvert W_r\rangle$ at an enhanced rate,
while the second step toward $\lvert rr\rangle$ is detuned by $V$. In the formal limit
$V\rightarrow\infty$, the last state can be projected away. At finite $V$, its amplitude
is suppressed only approximately, with a scale set by the ratio of coupling to detuning.
That amplitude may be small at every instant and still contribute a meaningful dynamical
phase after integration over the whole gate.

This same blockade has two very different computational uses. Here it mediates a coherent
two-qubit phase gate, so the complex phase accumulated along every return path matters.
In analog optimization, blockade can instead enforce a geometric exclusion rule: nearby
atoms are discouraged from being simultaneously excited. The companion tutorial on
[unit-disk mappings for neutral-atom arrays](/blog/2025/UnitDiskMapping/) develops that
second viewpoint. The hardware interaction is the same $C_6/R^6$ law, but the target is a
low-energy bit string rather than a phase-correct unitary. This distinction explains why
a pulse adequate for an analog independent-set experiment is not automatically a CZ
gate.

{% include figure.liquid path="assets/img/neutral-atom-control/part1-blockade-populations.png" alt="Two-panel scientific figure showing overlapping blockade volumes and the analytic-pulse populations from the computational state eleven" caption="Left: two projected blockade volumes overlap when the atoms are close enough. Right: the full finite-$V$ propagation from $\lvert11\rangle$ transiently occupies the symmetric single-Rydberg manifold and a smaller $\lvert rr\rangle$ component. Population return is not by itself a CZ test." %}

## What counts as CZ

An ideal controlled-Z is $\operatorname{diag}(1,1,1,-1)$, but a global pulse also gives
both qubits the same correctable $Z$ phase. The physically relevant target is therefore
the coset

$$CZ_\theta=\operatorname{diag}(1,e^{i\theta},e^{i\theta},-e^{2i\theta}).$$

This detail is not cosmetic. A reduced objective that observes too few relative phases
can announce success at a point that is not in this coset. That is precisely the gauge
hole encountered in the reduced Piccolo objective discussed in Part 3.

Leakage matters separately. The computational block $M=PUP$ need not be unitary when
population remains in a Rydberg state. The trace overlap used here penalizes both phase
error and missing amplitude, but later chapters still report leakage and independent
rollouts to avoid trusting a single scalar.

Four statements that are often blurred together should remain separate:

- **population return** asks whether each computational basis input comes back to some
  computational basis state;
- **survival** asks whether probability remains anywhere inside the computational
  subspace;
- **trace overlap** checks the complex computational block against a phase-selected
  target; and
- **entangling phase** checks the locally invariant phase combination that makes the
  diagonal action nonlocal.

The coherent gate studies use the squared computational-block trace overlap

$$
F_{\mathrm{tr}}=
\max_\theta\frac{
\left|\operatorname{tr}\!\left(CZ_\theta^\dagger PUP\right)\right|^2}{16}.
$$

Because $PUP$ is not assumed unitary, missing amplitude lowers this quantity. It is not,
however, automatically the same object as the process fidelity of a trace-preserving
channel. Part 4 will make that distinction explicit when dissipation is introduced.

### A phase-complete sanity check

Suppose a pulse returns all four computational basis states without mixing them, so its
projected action is approximately

$$
M=\operatorname{diag}
\left(a_{00}e^{i\phi_{00}},a_{01}e^{i\phi_{01}},
a_{10}e^{i\phi_{10}},a_{11}e^{i\phi_{11}}\right),
$$

where each $0\leq a_{jk}\leq1$ records endpoint survival for that input. Population
measurements see the $a_{jk}^2$ but are blind to the four phases. A global phase removes
one phase, and equal local $Z$ corrections remove the common single-qubit contribution.
What remains is the invariant combination

$$
\Phi=\phi_{00}-\phi_{01}-\phi_{10}+\phi_{11}.
$$

For a CZ, $\Phi=\pi$ modulo $2\pi$. If every amplitude equals one but every phase equals
zero, the population return is perfect and the operation is simply the identity: its
entangling phase is zero. Conversely, a pulse can have $\Phi\approx\pi$ while
$a_{11}<1$, meaning that it has generated the right conditional phase but has not
returned all population. These two failure directions motivate reporting both phase-aware
overlap and survival.

The target parameter $\theta$ in $CZ_\theta$ performs the allowed equal local-$Z$
adjustment. Maximizing over it is not permission to choose four arbitrary output phases;
the entire diagonal still has to lie on a one-parameter family. In a numerical
implementation, a useful sanity check is to compute $\Phi$ directly from the projected
diagonal after the optimizer finishes. Agreement between the optimized trace score,
endpoint survival, and $\Phi\approx\pi$ is much stronger evidence than any one of those
numbers alone.

## Why optimize

Analytic reasoning gives an excellent baseline. The Levine–Pichler construction uses two
constant segments at $\Omega_{\max}$, separated by a phase jump
$\xi=3.9024$, for a total duration of 683.2 ns
<d-cite key="levine2019parallel"></d-cite>. In a perfect-blockade model its recorded
infidelity is $5.3\times10^{-11}$. At finite blockade and $R=5\,\mu\mathrm m$, the
same pulse has infidelity $1.80\times10^{-2}$. The spacing scan is non-monotonic because
leakage amplitudes interfere; “stronger blockade is better” is only the envelope of the
story.

{% include figure.liquid path="assets/img/neutral-atom-control/part1-analytic-pulse-blockade.png" alt="Two-panel plot of the analytic Rabi quadratures and detuning, and CZ infidelity versus atom spacing" caption="Controls are shown as $\Omega/2\pi$ and $\Delta/2\pi$ in MHz. The pulse is exact only in the perfect-blockade approximation; the right panel uses the full finite-$V$ rollout." %}

Optimal control asks for a pulse that accounts for the actual Hamiltonian while balancing
several goals: high fidelity, short duration, limited amplitude, low bandwidth, small
leakage, and robustness. These goals conflict. A mathematically unconstrained optimizer
can exploit features that the device cannot emit; a heavily constrained optimizer can
return a slower but more physical answer.

The optimizer must therefore be told what “physical” means. A generic objective might
combine a terminal gate error, fluence, and Rydberg exposure,

$$
J[u,T]=1-F_{\mathrm{tr}}[U(T)]
+\alpha\int_0^T|\Omega(t)|^2dt
+\beta\int_0^T\langle n_r^{(1)}+n_r^{(2)}\rangle dt,
$$

while treating amplitude, endpoints, and bandwidth as constraints. The weights
$\alpha$ and $\beta$ express engineering priorities; they do not come from quantum
mechanics. Replacing a hard amplitude limit with a soft fluence term changes the
feasible set. Likewise, minimizing duration at a fixed fidelity is not equivalent to
maximizing fidelity at a fixed duration. Many apparent disagreements between methods
disappear once their actual optimization problems are written side by side.

### Translating an experimental request into an objective

Consider the apparently simple instruction “make the shortest high-fidelity CZ that the
laser can play.” It contains at least four mathematical decisions. “Shortest” suggests
minimizing $T$, but only after choosing a fidelity floor. “High fidelity” requires a
specific metric and a treatment of leakage. “The laser can play” may mean limits on peak
amplitude, quadrature slew, curvature, spectrum, or all four. Finally, the word “CZ” must
specify which local phases may be corrected later.

One formulation fixes $T$ and maximizes $F_{\mathrm{tr}}$. Repeating that problem over a
duration grid produces a numerical frontier, but each point is a separate optimization.
Another formulation minimizes $T$ subject to $F_{\mathrm{tr}}\geq F_\star$. This is closer
to the verbal request, yet it becomes harder because duration changes the dynamics and
the mesh. A third combines time and error in a weighted sum. That may be convenient, but
the chosen weight silently decides how much fidelity one nanosecond is worth.

Hardware limits are similarly sensitive to representation. The bound
$\sqrt{\Omega_x^2+\Omega_y^2}\leq\Omega_{\max}$ is a circular constraint on the complex
envelope. Independent boxes on $\Omega_x$ and $\Omega_y$ admit corner points with total
amplitude $\sqrt2\Omega_{\max}$. Penalizing fluence does not guarantee either bound.
Likewise, a smooth interpolation through jagged samples does not retroactively make the
optimized trajectory bandwidth-limited; it defines a new waveform that must be
propagated again.

The practical lesson is to write a short “control contract” before comparing solvers:
Hamiltonian and units, free controls, duration, target coset, fidelity convention,
initialization, hard constraints, soft penalties, and independent validation. Without
that contract, a table of final infidelities mixes physics choices with algorithmic
performance.

The finite-blockade scan also warns against calibrating from a single population trace.
A pulse can return $\lvert11\rangle$ with high probability but the wrong complex phase, or
obtain the correct entangling phase while leaving a small $\lvert rr\rangle$ amplitude. Gate
validation needs the complete computational action. In simulation that means
propagating four basis columns—or an equivalent phase-complete reduced model—and
checking both target overlap and survival.

## Costates and PMP

Start with a classical controlled system,

$$
\dot x=f(x,u,t),\qquad
J=\Phi(x(T))+\int_0^T L(x,u,t)\,dt.
$$

Introduce a costate $\lambda(t)$ and the control Hamiltonian
$\mathcal H=L+\lambda^\top f$. Pontryagin's maximum principle supplies the coupled
necessary conditions

$$
\dot x=\partial_\lambda\mathcal H,\qquad
\dot\lambda=-\partial_x\mathcal H,\qquad
0=\partial_u\mathcal H
$$

for an interior optimum, plus terminal and constraint conditions
<d-cite key="pontryagin1962mathematical"></d-cite>. The state moves forward from its
initial condition. The costate moves backward from information supplied by the terminal
objective. Their overlap tells the control how a local change affects the final cost.

For the sign convention $\mathcal H=L+\lambda^\top f$, a fixed initial state and free
terminal state give the transversality condition

$$
\lambda(T)=\nabla_x\Phi(x(T)).
$$

If final time is also free, an additional endpoint condition balances the control
Hamiltonian against any explicit terminal-time cost. Box-constrained controls replace
$\partial_u\mathcal H=0$ by a pointwise minimization or variational inequality; a
solution may live on the boundary. These details explain why simply clipping a gradient
step is not generally equivalent to solving the constrained PMP conditions.

In quantum control, complex variables may be split into real and imaginary components,
or handled with Wirtinger derivatives. For a unitary trajectory and a terminal gate
cost, the adjoint equation is still a backward Schrödinger-type propagation. The
terminal error seeds the costate, and the overlap of state, costate, and control
Hamiltonian produces the gradient. GRAPE is the time-discretized version of this loop;
direct collocation will instead expose all states and controls to a sparse nonlinear
program.

### Costate intuition from one small control change

Imagine changing $\Omega_x(t)$ only during a short interval around time $t_k$. The state
arriving at that interval contains everything the earlier pulse has done. The costate at
the same time contains, in reverse, how the terminal gate error values each possible
change leaving the interval. Their overlap through
$\partial H/\partial\Omega_x$ answers a local counterfactual: if this slice rotates the
current state a little more in the $x$ quadrature, does the final gate move toward or
away from the target?

This interpretation explains why the gradient needs two propagations rather than one.
A forward trajectory alone knows where the system is but not which deviations are useful
at the terminal time. A backward costate alone knows the terminal preference but not the
state on which a control generator acts. Multiplying the two pieces at every slice gives
all time-local derivatives after one forward and one backward sweep, avoiding a separate
finite-difference simulation for every control sample.

The terminal condition deserves equal attention. If the cost is
$\Phi(U(T))=1-F_{\mathrm{tr}}$, then the derivative of that phase-optimized overlap seeds
the matrix costate at $T$. If the target phase $\theta$ is optimized analytically, its
selected value must be held consistently while differentiating or treated as an
additional variable. If the terminal state is instead fixed as a hard equality,
transversality changes and a multiplier enforces that boundary. PMP provides necessary
conditions only after these endpoint choices are stated.

Constraints modify the stationarity picture as well. In the interior of an amplitude
disk, an optimum may satisfy a zero derivative. On the boundary, the unconstrained
gradient can point outward while the constrained optimum remains valid; a multiplier or
normal-cone condition balances it. This is why “compute an unconstrained gradient and
clip the result” is a heuristic, not a derivation of the constrained optimum. Later
chapters will contrast that heuristic with formulations that place bounds and smoothness
conditions directly inside the optimization problem.

{% include figure.liquid path="assets/img/neutral-atom-control/control-loop.png" alt="A control-flow diagram in which controls propagate the state forward, terminal cost initializes a backward costate, and the Hamiltonian gradient updates controls" caption="PMP is the continuous-time skeleton behind the forward/backward sweep used by GRAPE." %}

For a ket, $f(\psi,u)=-iH(u)\psi$. For a gate, the state may be the propagator $U$,
with $\dot U=-iH(u)U$. The quantum costate is another ket or matrix propagated backward.
Nothing mystical is added by the word “quantum”: unitarity gives a special dynamics and
gate fidelity gives a special terminal cost, but the variational logic is classical.

This also explains the family resemblance among the next methods. GRAPE discretizes the
forward/costate gradient. Krotov arranges a sequential update intended to guarantee
monotonic improvement under its assumptions. CRAB replaces the full time grid with a
small basis and lets a derivative-free optimizer move its coefficients. Collocation goes
further: it promotes the entire state history to an optimization variable and enforces
the dynamics as constraints.

All four methods still depend on the same scientific discipline: define a phase-complete
target, preserve units and basis order, and validate the returned pulse with a propagation
independent of the optimization bookkeeping. The algorithms differ in where dynamics
live and how the search space is shaped; none removes the need to decide what physical
success means. With that common foundation in place, the pulse shapes in Part 2 can be
read as consequences of parameterization and constraints rather than as mysterious
personalities of the solvers.

## Laboratory note

<div class="nac-lab-note"><strong>What changed.</strong> The first baseline looked almost solved in the perfect-blockade model. Restoring the actual $V=C_6/R^6$ interaction exposed a 1.8% error at the canonical spacing. That gap became the reason to optimize—not a desire to replace an elegant analytic pulse for its own sake.</div>

The ideal-blockade time-optimal reference from Jandura and Pupillo is 605.7 ns for these
amplitude units <d-cite key="jandura2022time"></d-cite>. It is a useful quantum speed
limit, not a promise that a finite-blockade, slew-limited, robust device pulse will reach
the same time. The rest of the series is largely the story of which qualifications must
be added to that number.

{% include neutral_atom_control/series_nav.liquid %}
