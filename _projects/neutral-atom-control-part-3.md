---
layout: distill
title: "3 · Optimizing the Whole Trajectory: Direct Collocation and Piccolo"
description: Dynamics defects, hard hardware constraints, minimum time, gauge freedom, and an honest Piccolo result
permalink: /projects/neutral-atom-control/part-3-collocation-piccolo/
tags: quantum-control collocation Piccolo CasADi
giscus_comments: false
importance: 99
category: work
show_on_projects: false
series: neutral-atom-control
series_part: 3
series_previous_url: /projects/neutral-atom-control/part-2-grape-krotov-crab/
series_previous_label: "Part 2"
series_next_url: /projects/neutral-atom-control/part-4-noise-robustness/
series_next_label: "Part 4"
status: draft
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: neutral-atom-control.bib
toc:
  - name: From shooting to transcription
  - name: Defects and constraints
  - name: The reduced CZ trajectory
  - name: What Piccolo adds
  - name: The gauge-hole experiment
  - name: Results and limits
  - name: Laboratory note
---

{% include neutral_atom_control/series_nav.liquid %}

## From shooting to transcription

GRAPE behaves like a shooting method: choose controls, simulate from $t=0$ to $T$,
measure the miss at the end, and adjust the controls. The intermediate states are
consequences of the control vector, not optimization variables.

Direct collocation changes the bookkeeping. Choose knot values
$\{x_k,u_k\}_{k=0}^{N}$ together, then require neighboring states to satisfy a
discretized dynamics relation. Schematically,

$$
\begin{aligned}
\min_{\{x_k,u_k\},T}\quad & T + R(u)\\
\text{subject to}\quad & d_k(x_k,x_{k+1},u_k,u_{k+1},\Delta t)=0,\\
& F(x_N)\ge F_{\min},\\
& g(x_k,u_k)\le0.
\end{aligned}
$$

The $d_k$ are _defects_. A nonlinear-programming solver sees terminal fidelity,
duration, dynamics, and hardware inequalities at once
<d-cite key="betts2010practical"></d-cite>.

{% include figure.liquid path="assets/img/neutral-atom-control/collocation.png" alt="A direct-collocation mesh in which state and control knots are simultaneous variables connected by dynamics-defect constraints" caption="The trajectory is no longer hidden inside a simulator. Amplitude, slew, curvature, endpoint, and fidelity conditions join the defects in one nonlinear program." %}

This is why tools such as Piccolo are useful. They are not “a better gradient” bolted onto
GRAPE. They provide a language for structured trajectories, multiple state and control
components, nonlinear dynamics, boundary conditions, path constraints, free time, and
sparse derivatives. Once a problem contains all of these, manually assembling the
nonlinear program becomes the dominant source of mistakes.

The Jacobian is sparse for a physical reason: defect $d_k$ touches only knots $k$ and
$k+1$. A terminal constraint touches the final state; a local amplitude bound touches
one control knot. Sparse automatic differentiation and a sparse interior-point solver
exploit this banded dependency graph. Transcription does not avoid simulation; it exposes
local structure in a form the nonlinear-programming solver can use.

This also changes initialization. A shooting method needs a control guess. Collocation
needs a plausible state trajectory as well. The workflow first solves an easier
fixed-duration or lower-fidelity problem, then uses that state/control history to warm start a harder
fidelity floor and minimum-time solve. Continuation in duration, interaction strength,
or constraint tightness is often more valuable than switching optimizers.

### A three-knot transcription

The mechanism is easiest to see on an intentionally tiny mesh. Let $x_0$ be fixed by the
initial quantum state and let $x_1,x_2$ and controls $u_0,u_1$ be decision variables. A
forward-Euler illustration would impose

$$
\begin{aligned}
d_0 &= x_1-x_0-\Delta t\,f(x_0,u_0)=0,\\
d_1 &= x_2-x_1-\Delta t\,f(x_1,u_1)=0.
\end{aligned}
$$

The actual quantum transcription uses a more accurate propagator defect, but the
dependency pattern is the same. Changing $x_1$ affects only $d_0$ and $d_1$; changing
$u_0$ affects only $d_0$. The terminal fidelity depends on $x_2$. This local coupling is
why the Jacobian and the second-derivative approximations have a narrow sparse structure.

At the first nonlinear-programming iteration, the guessed $x_1$ need not equal the state
obtained by simulating $u_0$. The mismatch appears as a nonzero defect. The solver moves
states and controls together to reduce defects while improving the objective and obeying
inequalities. At convergence, a small defect residual means the knots satisfy the chosen
discrete dynamics. It does not yet prove that the continuous Schrödinger equation between
knots follows the same path; that is the role of refinement and independent rollout.

Free duration can be handled by a normalized coordinate $\tau=t/T$. Then
$dx/d\tau=T f(x,u)$ and the interval size in physical time depends on $T$. This scaling
must appear in dynamics, slew, curvature, and accumulated costs. Treating $T$ as a label
while leaving a fixed $\Delta t$ in one of those expressions would optimize an internally
inconsistent problem.

## Defects and constraints

For linear-in-state quantum dynamics, a defect can be built from an approximation to the
propagator over one interval. This transcription uses midpoint controls and a Padé $(3,3)$ rational
approximation. A lower-order Padé $(2,2)$ approximation looked adequate in simpler
tests but was inaccurate here because the interaction phase per step was not tiny:
$V\Delta t\approx0.55$.

That is a subtle collocation failure mode. The optimizer can satisfy the _discrete_
defects extremely well while the continuous equation follows a different trajectory.
Every candidate is therefore replayed with an independent matrix-exponential rollout.
The reported $F_{\mathrm{rollout}}$, rather than only the NLP's
$F_{\mathrm{discrete}}$, is the final check.

The constrained trajectory imposes:

- the circular amplitude bound
  $\Omega_x^2+\Omega_y^2\le\Omega_{\max}^2$;
- zero control at both endpoints;
- a slew limit of $148$ rad/µs²;
- a curvature limit of $1739$ rad/µs³;
- a hard lower bound on terminal CZ fidelity; and
- free final time in the minimum-time runs.

These are natural inequality and boundary constraints in collocation. In a conventional
GRAPE implementation they would require projections, penalties, filtering, or a
reparameterization—and each choice changes the landscape.

On a uniform mesh, the path constraints become explicit algebra:

$$
\begin{aligned}
\Omega_{x,k}^2+\Omega_{y,k}^2 &\le \Omega_{\max}^2,\\
\|u_{k+1}-u_k\|_2 &\le s_{\max}\Delta t,\\
\|u_{k+1}-2u_k+u_{k-1}\|_2 &\le c_{\max}\Delta t^2,\\
u_0=u_N&=0.
\end{aligned}
$$

The factors of $\Delta t$ matter when final time is free. Omitting them changes the
physical slew or curvature bound whenever the optimizer changes $T$. Penalties are not
equivalent to these inequalities: a penalty permits violation in exchange for objective
improvement, whereas a hard constraint defines the feasible set.

Each inequality removes a different kind of unphysical shortcut. The amplitude disk
limits instantaneous optical power and, unlike independent component boxes, constrains
the magnitude of the complex envelope directly. A slew limit restricts how far that
envelope can move during one unit of physical time. A curvature limit restricts how
abruptly the slew itself can change, discouraging sharp corners even when the first
difference remains legal. Endpoint constraints reserve time to turn the field on and off
rather than allowing a discontinuity at the gate boundary.

These constraints should be interpreted on the variables they actually bind. Bounding
the optimizer samples does not guarantee that a cubic interpolant remains inside the
amplitude disk between knots. Bounding finite differences does not by itself impose the
frequency response of an acousto-optic modulator. Conversely, a device transfer function
may smooth a waveform without satisfying the same mathematical curvature bound. The
purpose of the collocation constraints is to define a controlled intermediate model, not
to claim that one mesh has already captured the complete optical chain.

The active-constraint plot helps diagnose which requirement sets the solution. A ratio
near one means the trajectory is using nearly all of that resource. Here amplitude is
close to active over part of the gate, while the saved slew and curvature ratios retain
more margin. If a minimum-time solution never approaches any path bound, duration may be
limited instead by the quantum dynamics or by the terminal fidelity condition. If many
constraints are active simultaneously, small changes in their numerical values can move
the optimum appreciably and continuation becomes especially useful.

### What the nonlinear solver sees

An interior-point solver does not reason in terms of lasers or Rydberg atoms. It sees a
large vector of real decision variables, a scalar objective, equality residuals, and
inequality residuals. Barrier terms keep iterates away from forbidden inequality
boundaries while Newton-like linear systems couple primal variables and constraint
multipliers. Sparse derivatives make those systems tractable; poor scaling can make them
appear singular even when the physics is well posed.

Scaling deserves explicit design. State amplitudes are naturally order one, controls are
tens of rad/µs, curvature bounds may be thousands of rad/µs³, and infidelity can be
$10^{-4}$ or smaller. Feeding these raw magnitudes into one system can cause the solver
to emphasize some residuals merely because their numbers are larger. Normalizing controls
and time, or assigning constraint scales, improves numerical conditioning without
changing the physical feasible set.

Solver status is evidence, not a verdict. “Optimal” means the discretized first-order
conditions meet configured tolerances. “Maximum iterations” may still leave a useful
feasible candidate, while “infeasible” may indicate a poor initialization or a genuinely
empty constraint set. The handoff should record terminal score, maximum defect, maximum
path violation, complementarity or stationarity diagnostics, and independent rollout.
Reporting only the solver's final word hides the information needed to reproduce or
challenge the result.

Constraint multipliers can be scientifically informative. A large multiplier on the
fidelity floor estimates how much the objective would improve if that floor were relaxed
slightly; a large amplitude multiplier marks intervals where more optical authority would
reduce time. These are local sensitivities, not global design laws, but they help decide
whether the next experiment should change a hardware bound, refine the mesh, or search
from another seed.

{% include figure.liquid path="assets/img/neutral-atom-control/part3-active-constraints.png" alt="Three aligned plots showing amplitude, slew, and curvature of the collocation fidelity 0.9999 pulse as fractions of their imposed bounds" caption="The saved collocation pulse is shown on its knot grid without smoothing. Amplitude nearly uses its available disk while slew and curvature retain margin. Normalizing by the actual bounds makes active and inactive constraints directly visible." %}

Mesh feasibility is still not continuous-time feasibility. Linear interpolation can
overshoot a nonlinear derived quantity between knots, and an approximate defect can hide
integration error. The audit therefore checks endpoint values, discrete inequalities,
Hamiltonian Hermiticity, and a separate piecewise rollout. Mesh refinement should
stabilize both the objective and the independently propagated gate.

### Discrete success versus physical success

There are three residuals worth separating. The nonlinear-programming residual measures
how well the decision variables satisfy the chosen defects and constraints. The rollout
discrepancy measures how far a trusted integrator lands from the optimized terminal
state when driven by the saved controls. The model discrepancy measures how far either
trajectory is from the real system. Tightening the solver tolerance addresses only the
first. Raising the Padé order or refining the mesh addresses the second. Improving
calibration and the Hamiltonian addresses the third.

The observed $V\Delta t\approx0.55$ explains why propagator accuracy matters here. The
interaction generates an appreciable phase within one mesh interval. A low-order defect
can be smooth and differentiable enough for the optimizer to exploit while still
accumulating a biased phase across many intervals. An independent exponential rollout is
valuable precisely because it shares the Hamiltonian but not the rational approximation
used inside the nonlinear program.

Warm starts should also be validated rather than trusted. A state guess constructed by
rolling out the initial controls begins dynamically consistent, which usually reduces
early defect work. After changing duration or interaction strength, resampling the old
trajectory may introduce defects; it is still useful, but the solver must repair them.
Continuation can follow one local family past several fidelity floors, so multiple seeds
or reverse continuation provide a check that a surprisingly favorable point is not only
an artifact of one path through parameter space.

## The reduced CZ trajectory

The full two-qutrit gate propagator has 81 complex entries. Global exchange symmetry and
the specific target mean we do not need all of them. The reduced formulation follows the driven amplitudes
needed to reconstruct the computational action: two amplitudes for a single-excitation
block and three for the symmetric double-excitation block. Splitting real and imaginary
parts gives ten real state variables per knot.

Reduction is powerful, but it must preserve the target's phases. A good reduced state is
not merely one that reproduces populations. It must contain every observable needed to
distinguish the desired gate from a gauge-equivalent-looking impostor.

## What Piccolo adds

Piccolo.jl packages quantum trajectory optimization around this transcription viewpoint
<d-cite key="piccolo119"></d-cite>. Its abstractions can represent ket, unitary, and
multi-component trajectories; objectives and constraints are assembled around a direct
collocation solve. QuantumCollocation.jl's development was folded into Piccolo, so the
relevant question is not whether the old repository name survives—it is whether the
current trajectory and objective encode the physical target.

The earlier discussion of this CZ formulation is preserved in the
QuantumCollocation issue tracker <d-cite key="quantumcollocation88"></d-cite>. The
results here refer specifically to Piccolo v1.19.0 rather than claiming that every later
version or objective encoding has the same behavior.

We tested Piccolo v1.19.0 on progressively reduced versions of this problem. The results
are deliberately reported as a tool study, not edited into a success story:

| Piccolo formulation           | Outcome                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| raw 9-level unitary, 61 knots | stopped after more than 30 minutes                                                   |
| reduced 5-level CZ            | fast, but gauge-blind; physical gate fidelity about 0.79                             |
| phase-anchored 6-level CZ     | correct objective, about 22 s/iteration; about 0.63 after 300 unconverged iterations |
| Bell-state ket trajectory     | $F_{\mathrm{rollout}}=0.999992$ in about 1.005 µs                                    |

The Bell result demonstrates that the installed Piccolo stack and trajectory machinery
work on this neutral-atom Hamiltonian. It does **not** demonstrate a converged Piccolo CZ.

That distinction separates software capability from objective correctness. A trajectory
package can provide state representations, automatic differentiation, defect templates,
constraint composition, solver interfaces, and visualization. Those facilities reduce
the amount of custom numerical plumbing and make it easier to state coupled trajectory
problems. They cannot infer that a diagonal gate must preserve a particular invariant
phase if the supplied terminal objective never asks for it.

The Bell-state test is useful because its terminal question is narrower: does one chosen
input reach one chosen target ket up to a global phase? A full gate test asks for the
correct action on an entire computational subspace, including relative phases among
columns. Success on state transfer confirms dynamics and optimization machinery, but it
does not validate a gate objective. This hierarchy of tests is healthy: begin with a
single trajectory, then add phase references and basis columns, and independently score
the full map after each increase in abstraction.

CasADi/IPOPT and Piccolo occupy different layers in this comparison. CasADi supplies
symbolic expressions, sparse derivatives, and a general nonlinear-programming backend;
the lower-level formulation assembles the reduced quantum defects and constraints explicitly. Piccolo
supplies quantum trajectory types and higher-level objective/constraint abstractions
before invoking its optimization stack. The latter can reduce formulation code,
especially for multiple coupled trajectories, but abstraction does not remove the need
to inspect the physical terminal map.

| Question       | GRAPE-style loop                                 | Direct collocation / Piccolo-style trajectory        |
| -------------- | ------------------------------------------------ | ---------------------------------------------------- |
| state history  | generated by each simulation                     | explicit decision variables                          |
| dynamics       | satisfied by the chosen integrator               | sparse defect constraints                            |
| path limits    | projection, penalty, filter, or parameterization | algebraic inequalities at knots                      |
| free duration  | outer loop or rescaled dynamics                  | variable with consistently scaled constraints        |
| failure signal | poor terminal objective or bad rollout           | infeasibility, residuals, or bad independent rollout |

Neither column is universally better. A waveform search is compact and often fast when
only terminal fidelity matters. A trajectory formulation earns its complexity when
state-dependent constraints, free time, multiple models, or hardware inequalities are
central to the question.

## The gauge-hole experiment

Why did the fast five-level reduction fail? Suppose the reduced state checks the
single-excitation return amplitude and a double-excitation amplitude but omits one phase
reference. The optimizer can make all observed magnitudes and selected phase differences
look correct while changing the invariant entangling phase

$$\phi_{\mathrm{ent}}=\arg U_{11}-\arg U_{10}-\arg U_{01}+\arg U_{00}.$$

A local $Z$ correction changes individual phases but not this combination. The target
requires $\phi_{\mathrm{ent}}=\pi$ modulo $2\pi$. If the reduced objective cannot
reconstruct it, an apparent optimum need not be a CZ-coset point.

Adding the missing phase anchor repairs the functional. It also makes the optimization
heavier. That trade is unavoidable: a fast objective that cannot distinguish success
from failure is not an approximation to the problem.

### Following the phases by hand

For a diagonal two-qubit operation, local phase corrections transform the basis phases
in a structured way:

| basis state | original phase | after equal local $Z(\theta)$ corrections |
| ----------- | -------------- | ----------------------------------------- |
| $\lvert00\rangle$ | $\phi_{00}$ | $\phi_{00}$ |
| $\lvert01\rangle$ | $\phi_{01}$ | $\phi_{01}+\theta$ |
| $\lvert10\rangle$ | $\phi_{10}$ | $\phi_{10}+\theta$ |
| $\lvert11\rangle$ | $\phi_{11}$ | $\phi_{11}+2\theta$ |

Substituting the corrected phases into
$\Phi=\phi_{00}-\phi_{01}-\phi_{10}+\phi_{11}$ cancels both copies of $\theta$.
A global phase cancels for the same reason. Thus $\Phi$ labels information that neither
global nor allowed local rephasing can repair.

As a concrete example, phases $(0,0,0,\pi)$ give $\Phi=\pi$ and represent CZ. Phases
$(0,\pi/3,\pi/3,2\pi/3)$ give $\Phi=0$: every basis state may return with unit
population, and the single-excitation phases even look symmetric, but the operation is
locally equivalent to a nonentangling diagonal gate. An objective observing only return
magnitudes and the equality $\phi_{01}=\phi_{10}$ cannot distinguish these cases.

A phase anchor adds one complex reference or an equivalent invariant constraint so that
the reduced variables reconstruct $\Phi$. That extra information couples terminal
conditions that were previously independent and can worsen optimization conditioning.
The increased cost is evidence that the corrected problem is richer, not a reason to
prefer the faster gauge-blind answer.

{% include figure.liquid path="assets/img/neutral-atom-control/part3-gauge-hole.png" alt="Phase-vector schematic showing that returned populations can carry incorrect phases and a curve of phase-aware CZ score versus invariant entangling phase" caption="A population-correct diagonal trajectory still has a gauge-invariant entangling phase. Local single-qubit $Z$ phases cannot change $\Phi=\phi_{00}-\phi_{01}-\phi_{10}+\phi_{11}$; the objective must retain enough phase references to test it." %}

For a diagonal computational action, the invariant is transparent. A common global
phase cancels, and equal local phases on the atoms shift the four basis phases in a
pattern that also cancels from $\Phi$. The CZ coset requires $\Phi=\pi$ modulo $2\pi$.
A reduced trajectory that records only return magnitudes, or anchors too few complex
amplitudes, leaves $\Phi$ free. The optimizer then uses that unpenalized direction
because it makes the remaining constraints easier.

## Results and limits

{% include figure.liquid path="assets/img/neutral-atom-control/part3-collocation-piccolo-results.png" alt="Two-panel figure showing minimum collocation duration versus fidelity floor and the successful Piccolo Bell-state controls" caption="Left: independent rollout of the collocation solutions meets the requested CZ fidelity floors at approximately 1042, 1069, and 1077 ns. Right: the smooth Piccolo Bell trajectory is a state-transfer validation, not a CZ solution. Controls are divided by $2\pi$ and shown in MHz." %}

The constrained minimum-time solutions are:

| Fidelity floor |  Duration | Independent rollout |
| -------------: | --------: | ------------------: |
|           0.99 | 1041.6 ns |             0.99009 |
|          0.999 | 1068.8 ns |             0.99903 |
|         0.9999 | 1076.9 ns |             0.99991 |

These gates are slower than the 605.7 ns ideal-blockade reference and the 683.2 ns
analytic pulse. That is expected: finite blockade, zero endpoints, slew, curvature, and
a hard rollout-validated fidelity floor define a stricter problem. “Slower” is not solver
failure when the feasible sets are different.

The left panel also shows an important practical pattern: increasing the requested
fidelity by two nines costs only about 35 ns within this constrained family. The dominant
time is spent entering and leaving the admissible pulse smoothly, not polishing the last
decimal in an unconstrained landscape.

“Minimum time” is local to this transcription, initialization, mesh, and constraint
set. It is not a proof of a global quantum speed limit. The sound comparison is between
independently rolled-out feasible points under the same constraints. The 605.7 ns
Jandura–Pupillo solution remains the ideal-blockade reference; this experiment answers a stricter
finite-blockade, zero-endpoint, slew- and curvature-limited question.

A stronger minimum-time claim would require systematic mesh refinement, multiple seeds,
consistent constraint scaling, and evidence that further continuation does not find a
shorter feasible branch. Even then it would be numerical evidence for the stated model,
not an analytic bound. The value of the present frontier is more modest and more useful:
it quantifies the duration cost of a particular set of physical constraints and supplies
fully rolled-out candidates for the subsequent noise comparison.
It also leaves a reproducible baseline against which a different mesh, objective, or
trajectory package can be compared without silently changing the scientific question.

## Laboratory note

<div class="nac-lab-note"><strong>What failed.</strong> Piccolo's fast reduced CZ solve optimized the objective it was given. The objective had forgotten a physical phase, so the resulting gate was wrong. A phase-anchored formulation fixed the science but exposed a performance problem. The honest result is therefore one constrained CZ success in the CasADi transcription, one high-fidelity Piccolo Bell trajectory, and no converged Piccolo CZ.</div>

The episode answers the broader “why a trajectory tool?” question with a qualification.
Piccolo makes rich formulations easier to express, but no library can decide which
phases define the user's gate. Modeling and validation remain part of optimal control.

The next chapter changes the dynamics again. Once dissipation enters, “coherent
infidelity” is no longer the quantity that ranks the pulses we already found.

{% include neutral_atom_control/series_nav.liquid %}
