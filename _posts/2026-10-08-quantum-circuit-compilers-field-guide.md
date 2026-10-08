---
layout: distill
title: "A field guide to quantum circuit compilers"
description: What quantum compilers and transpilers do to a circuit, how they decide one circuit is better than another, the main families of methods and tools, and where research is heading, including compilers for fault-tolerant machines. Part 1 of a series on quantum circuit optimization.
tags: quantum-computing circuit-optimization compilers
categories: quantum-computing
giscus_comments: false
date: 2026-10-08 10:00:00 +0800
featured: false
related_posts: true
bibliography: 2026-10-quantum-compilers.bib
og_image: https://nez0b.github.io/assets/img/quantum_circuit_optimization/field-map.png
series: quantum-circuit-optimization
series_part: 1
series_next_url: /blog/2026/classiq-logo-phase-oracle-depth-95/
series_next_label: "Part 2: A depth-95 phase oracle for the Classiq logo"

authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research

toc:
  - name: From a program to a runnable circuit
    subsections:
      - name: Borrowed from classical compilers
      - name: Inside one transpiler
  - name: What counts as better
  - name: Building a circuit from a specification
    subsections:
      - name: Two qubits are solved
      - name: Larger unitaries
      - name: Structured inputs
      - name: Oracles from Boolean functions
  - name: Improving a circuit you already have
    subsections:
      - name: Local rewrites
      - name: Resynthesis
      - name: Changing the language
      - name: Search and learning
  - name: Layout, routing and scheduling
  - name: The tools, and how to trust them
    subsections:
      - name: The main tools
      - name: Checking the output
      - name: Benchmarks
  - name: The fault-tolerant frontier
    subsections:
      - name: Rotations become T gates
      - name: Fewer T gates
      - name: Lattice surgery and magic states
      - name: A resource estimate is the final score
  - name: What gate-level passes cannot see
  - name: Further reading

_styles: >
  .qco-series-nav {
    margin: 0 0 1.75rem;
    padding: 0.8rem 1rem;
    border: 1px solid var(--global-divider-color);
    border-radius: 0.5rem;
    background: var(--global-card-bg-color);
    font-size: 0.92rem;
  }
  .qco-series-kicker {
    /* muted, but above 4.5:1 in both themes; the theme's light grey is 3.8:1 on white */
    color: color-mix(in srgb, var(--global-text-color) 72%, var(--global-bg-color));
  }
  .qco-series-links {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.4rem 1rem;
    margin-top: 0.35rem;
  }
  .qco-lede {
    font-size: 1.08rem;
    line-height: 1.75;
  }
  @media (min-width: 1025px) {
    d-article d-contents {
      grid-row: auto / span 6;
    }
  }
  /* distill's caption grey is black at 60%, which nearly vanishes on the dark theme; mirror it as white at 60% */
  html[data-theme="dark"] d-article figcaption.caption {
    color: rgba(255, 255, 255, 0.6);
  }
  d-article mjx-container[display="true"] {
    overflow-x: auto;
    overflow-y: hidden;
    max-width: 100%;
  }
  d-article table {
    display: block;
    overflow-x: auto;
  }
  d-article table td:first-child,
  d-article table th:first-child {
    white-space: nowrap;
  }
---

{% include quantum_circuit_optimization/series_nav.liquid %}

<p class="qco-lede">
A quantum compiler takes the circuit you wrote and returns one that a particular machine can run, usually with fewer gates. This post is a field guide to what happens in between, for readers who know what qubits, gates and circuit diagrams are but have never looked inside a compiler.
</p>

I wrote it while working on the Classiq Quantum Circuit Challenge, the subject of [Part 2](/blog/2026/classiq-logo-phase-oracle-depth-95/). The task was a _phase oracle_, a circuit that puts a minus sign on the basis states that stand for the black pixels of the Classiq logo. When I ran the best off-the-shelf compilers on the challenge's baseline circuit, they made it about 6 percent shorter, while a redesign of the same oracle was 56 times shorter. The reason is that a compiler must keep the circuit's behaviour on every input and is never told the facts that would allow a much shorter circuit, as [a later section](#what-gate-level-passes-cannot-see) explains. Before that come the jobs a compiler does, the costs it optimises, the main families of methods, the tools you can install today, and where research is going, including compilers for fault-tolerant machines, which count costs very differently.

{% include figure.liquid path="/assets/img/quantum_circuit_optimization/field-map.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A tree diagram: quantum compiler branches into six jobs: translate into native gates, synthesise from a specification, improve a circuit, map to the hardware, check the result, and a fault-tolerant back end. Each job has one or two leaves naming its main methods and tools." caption="<strong>The jobs of a quantum compiler.</strong> A compiler is several kinds of work, each with its own methods and tools. The sections below take them roughly in this order." zoomable=true %}

## From a program to a runnable circuit

### Borrowed from classical compilers

A classical compiler reads source code, turns it into an _intermediate representation_ (IR) that is easy to analyse, runs a sequence of _passes_ that each rewrite or analyse the IR, and finally emits instructions for one target processor. Quantum toolchains copied this shape. The paper describing Qiskit calls its design "similar to classical compiler infrastructures such as LLVM" <d-cite key="javadiabhari-qiskit-2024"></d-cite>: a circuit is held as a graph of gates, and a pass manager runs passes over it in stages.

The IR is usually the circuit itself, as a list or a dependency graph of gates. The common text format for exchanging circuits is OpenQASM. Its third version adds classical control flow and timing, so a program can measure a qubit mid-circuit and branch on the result <d-cite key="cross-openqasm3-2022"></d-cite>. Compiling such _dynamic circuits_ is a research topic of its own <d-cite key="niu-acdc-2024"></d-cite>. Other IRs borrow directly from classical compilers; QIR, for example, builds on the LLVM infrastructure <d-cite key="stade-qir-2025"></d-cite>. A 2025 review compares these IRs <d-cite key="cardama-ir-review-2025"></d-cite>.

You will also meet the word _transpiler_. Qiskit uses it for the part of the compiler that takes a circuit and returns a circuit, as opposed to the part that turns a high-level program into a first circuit. Most of this post is about transpilers, because that is where most optimisation happens.

The target matters at every step. A real device runs only a few _native_ gates, typically one kind of two-qubit gate (CZ, for example) plus a handful of one-qubit rotations, and on most superconducting chips a two-qubit gate is possible only between physically neighbouring qubits. Gate errors, coherence times and gate durations differ from qubit to qubit and are recalibrated often.

### Inside one transpiler

The [Qiskit documentation](https://quantum.cloud.ibm.com/docs/en/guides/transpiler-stages) lists six stages for Qiskit's preset pass managers, which I also read off the installed library (Qiskit 2.5.2):

{% include figure.liquid path="/assets/img/quantum_circuit_optimization/compile-pipeline.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A vertical pipeline of six stages from an input circuit to a device-ready circuit: init, layout, routing, translation, optimization and scheduling, each with a one-line description. Layout and routing are bracketed as existing only because not every pair of qubits is wired; the optimization stage has a loop arrow labelled repeat." caption="<strong>The stages of Qiskit's transpiler.</strong> Each stage has one job. Layout and routing exist only because the device does not connect every pair of qubits. The optimization stage repeats its passes while depth and gate count keep falling. Scheduling does not run unless you ask for it." zoomable=true %}

Only one stage is called optimization, but every stage makes choices that change the result. Layout picks which physical qubit holds each of the program's qubits. Routing then inserts SWAP gates, which exchange the states of two neighbouring qubits, until every two-qubit gate acts on neighbours. A poor layout forces routing to add many SWAPs, and no later pass removes all of them. The preset pipelines also come in optimization levels, from 0 to 3, which trade compile time for circuit quality; level 3 tries more passes and more layouts. Compile time is a real cost for large circuits and for workflows that compile many circuits.

tket organises its compiler differently but does the same jobs <d-cite key="sivarajah-tket-2021"></d-cite>, and so do the other tools in the table further down.

## What counts as better

A compiler needs a number to minimise, and there are several candidates.

- The two-qubit gate count. On current hardware two-qubit gates have error rates several times higher than one-qubit gates, so their number is the most common proxy for the error a circuit accumulates.
- Depth, the number of layers when gates on different qubits run in parallel. Qubits lose their state while they wait, so a shorter circuit loses less. Some papers count only two-qubit layers.
- The estimated success probability, the product of the calibrated fidelities of every gate used. This lets the compiler prefer good qubits and avoid bad ones <d-cite key="murali-noiseadaptive-2019"></d-cite>.
- On a fault-tolerant machine, the number of T gates (T-count), the number of layers that contain them (T-depth), and finally the physical qubits and hours the whole computation needs <d-cite key="litinski-gameofsurfacecodes-2019"></d-cite>. The T gate is a rotation by $$\pi/4$$ about $$Z$$. On an error-corrected machine each T gate consumes a separately prepared _magic state_, which has long made it the most expensive gate in resource estimates; the [fault-tolerant section](#the-fault-tolerant-frontier) explains why, and how that is changing.
- Compile time, which bounds how hard the compiler can search.

These numbers can point in different directions. I ran three tools on the baseline circuit from Part 2, an 18-qubit circuit with depth 5329 and 3502 CNOT gates (CX in the figures). I used my own reproduction of it, which scores the same, allowed every pair of qubits to interact, and checked every output for correctness:

{% include figure.liquid path="/assets/img/quantum_circuit_optimization/cost-models.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Three bar charts side by side, for depth, CX count and T-count, each with six rows: the baseline circuit, one-qubit gates merged only, Qiskit level 3, tket then Qiskit, PyZX teleport_reduce and PyZX full_reduce. tket then Qiskit is best for depth (5004) and CX (3347); the two PyZX rows tie for the lowest T-count (3167 and 3160, within the counter's tolerance), and full_reduce is worst for depth (8950) and CX (8576)." caption="<strong>Which tool wins depends on the cost.</strong> Six versions of one 18-qubit circuit, all checked correct. Qiskit and tket lower depth by about 6 percent and the CNOT count by about 4. Both PyZX methods lower the T-count by a quarter; full_reduce also more than doubles the CNOTs. Merging neighbouring one-qubit gates, which every tool does, accounts for much of the T-count drop of the first rows. Versions: Qiskit 2.5.2, pytket 2.18.1, PyZX 0.10.6." zoomable=true %}

Qiskit at level 3 and tket followed by Qiskit improve depth and CNOT count a little. PyZX, a tool based on the ZX-calculus described below, cuts the T-count from 4247 to about 3160 with either of two methods, and one of them, full_reduce, pays for it with 8576 CNOTs instead of 3502. Under a cost model that counts only T gates, the two PyZX results are the best of the six; under one that counts CNOTs, full_reduce is the worst. For today's noisy hardware the tket result is the one to run. PyZX counts every phase that is not a multiple of $$\pi/2$$ as one T gate. Eleven of the circuit's rotations have angles that are not multiples of $$\pi/4$$, and a fault-tolerant compiler would expand each of those into many T gates, so compiled for such a machine every circuit here would need more T gates than the chart shows.

## Building a circuit from a specification

Sometimes the input is a description of what the circuit should do: a matrix, a state to prepare, a Boolean function, or a list of Pauli rotations from a Hamiltonian. Turning that into gates is called _synthesis_. It is also a tool inside optimisation, because a block of an existing circuit can be cut out, described as a matrix, and synthesised again.

### Two qubits are solved

Any two-qubit gate can be built from at most three CNOTs and some one-qubit gates, and some two-qubit gates need all three <d-cite key="vatan-twoqubit-2004,vidal-threecnot-2004,shende-minimal2q-2004"></d-cite>. Because of this, a transpiler can collect any run of gates that acts on the same two qubits, multiply them into one $$4\times4$$ matrix, and rebuild the block with at most three CNOTs. Qiskit does this in its optimization stage.

### Larger unitaries

For $$n$$ qubits, exact decompositions of an arbitrary unitary exist, but the number of CNOTs they need grows like $$4^n$$ <d-cite key="shende-synthesis-2006"></d-cite>, so they are useful only for a few qubits. In that range, numerical synthesis often finds much shorter circuits. QSearch, for example, grows a circuit structure one two-qubit gate at a time and, for each structure, fits the one-qubit angles by numerical optimisation until the circuit matches the target <d-cite key="davis-qsearch-2020"></d-cite>. BQSKit packages methods of this kind and scales them to large circuits by cutting the circuit into small blocks and resynthesising each one <d-cite key="younis-bqskit-2021,kukliansky-qfactor-2023"></d-cite>.

Numerical synthesis also makes _approximate_ compilation natural. If the target is a distance $$\varepsilon$$ instead of exact equality, a shorter circuit is often possible. Madden and Simonetto compress a standard decomposition of arbitrary unitaries by a factor of two "without practical loss of fidelity" <d-cite key="madden-aqc-2022"></d-cite>. On noisy hardware a slightly wrong but much shorter circuit can even give better answers than an exact long one, and QUEST exploits this by running several different approximations of one circuit and combining their outputs <d-cite key="patel-quest-2022"></d-cite>.

### Structured inputs

When the specification has structure, a method that knows the structure does far better than generic unitary synthesis.

- Clifford circuits. Circuits built from H, S and CNOT can be synthesised with provably minimal depth by SAT solvers for small sizes <d-cite key="peham-cliffordsat-2023"></d-cite>.
- Pauli rotations. Simulating a Hamiltonian turns into a list of rotations $$e^{-i\theta P}$$ where $$P$$ is a product of Pauli operators. Synthesising the list as a whole, choosing the order and sharing CNOTs between neighbouring rotations, beats compiling each rotation alone; greedy methods by Goubault de Brugière and Martiel report depth up to four times shorter than earlier heuristics <d-cite key="goubaultdebrugiere-rustiq-2024"></d-cite>. QuCLEAR reports up to 77.7 percent fewer CNOTs by moving Clifford gates to the end of the circuit and absorbing them into the measurement and classical post-processing <d-cite key="liu-quclear-2025"></d-cite>. The result computes the same expectation values but is no longer the same unitary.
- States. Preparing a given state from $$\lvert 0\dots0\rangle$$ is easier than building a whole unitary. With enough helper qubits, any $$n$$-qubit state can be prepared in depth proportional to $$n$$, although the number of helpers grows exponentially <d-cite key="zhang-state-prep-depth-2022"></d-cite>.

### Oracles from Boolean functions

Many algorithms need an _oracle_: a circuit that computes a classical function $$f(x)$$ of the input bits, either into an extra qubit or as a sign on each basis state, as in Grover's algorithm. Part 2 builds one. Computing $$f$$ usually needs intermediate results, which go on helper qubits (_ancillas_) that start at 0. Once the answer has been used, the circuit runs the same steps backwards to return the helpers to 0, which is called _uncomputation_. Without it the helpers stay entangled with the input and spoil the interference the algorithm relies on.

Logic synthesis, borrowed from classical chip design, supplies the forms. One writes $$f$$ as an exclusive sum of products (ESOP), an XOR of AND terms, each of which becomes a multi-controlled NOT gate <d-cite key="meuli-esop-2019"></d-cite>. Another breaks $$f$$ into small lookup tables with classical logic-synthesis tools and builds a reversible gate for each <d-cite key="soeken-lhrs-2019"></d-cite>. A third writes $$f$$ as a network of XOR and AND gates. XORs cost only CNOTs, and a circuit for $$f$$ needs at most four T gates and one helper qubit per AND gate, so for a fault-tolerant target the number of ANDs matters most <d-cite key="meuli-multiplicative-complexity-2019"></d-cite>. A Toffoli gate that is correct except for extra phases on some inputs is cheaper still, and the phases cancel when the same gate is undone during uncomputation <d-cite key="maslov-rtof-2016"></d-cite>. Part 2's oracle has this shape: helpers computed and later uncomputed, around a middle that is an XOR of AND terms.

## Improving a circuit you already have

Most of what a transpiler calls optimisation takes a correct circuit and returns a shorter one that implements the same unitary. The methods fall into four families, which differ in how much of the circuit they look at.

{% include figure.liquid path="/assets/img/quantum_circuit_optimization/improve-families.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Four rows, each with a before and after example. 1, local rewrites: Rz alpha, CX, Rz beta, CX becomes Rz of alpha plus beta. 2, re-synthesise a block: a two-qubit block with four CX becomes a block with three CX. 3, change the language: a graph of eight green and red ZX nodes becomes a graph of three. 4, search over rewrites: a tree of circuit costs where a step from 12 to the worse 13 leads to 8 and then 7, while always taking the best next step stops at 10." caption="<strong>Four ways to improve a circuit.</strong> They differ in how far they look: one pair of gates, one small block, the whole circuit in another notation, or a search over many rewrites. Row 4's costs are made up to show why a search sometimes accepts a worse circuit." zoomable=true %}

### Local rewrites

The oldest methods are _peephole_ rewrites borrowed from classical compilers. Two CNOTs in a row cancel, two rotations about the same axis merge into one, and a gate can often move past another (a $$Z$$ rotation commutes with the control of a CNOT) to bring a cancelling pair together. Nam and co-authors built a fast optimiser for large circuits from rules of this kind, including a rotation-merging pass that finds rotations far apart in the circuit that act on the same parity, the XOR of the same set of qubit values <d-cite key="nam-heuristic-2018"></d-cite>. Template matching generalises the idea: it searches the circuit for any subcircuit equal to part of a known identity and replaces it with the cheaper remainder <d-cite key="iten-templatematching-2022"></d-cite>.

Local rewrites are fast and safe, and they are the bulk of every optimisation stage. They also stop early, at the first circuit where no single rule applies.

### Resynthesis

The second family cuts out a block and synthesises it again from its matrix. The two-qubit case above is the everyday example. BQSKit does the same for blocks of three or four qubits with numerical synthesis, which finds savings no rule list contains but costs much more time <d-cite key="younis-instantiation-2022,nation-benchpress-2025"></d-cite>.

### Changing the language

The third family translates the circuit into a different notation, simplifies it there, and translates back. The best-known example is the _ZX-calculus_, a graphical language in which a circuit becomes a graph of green and red nodes ("spiders") carrying phases <d-cite key="coecke-zx-2011"></d-cite>. The graph obeys rewrite rules that circuits do not have, so it can shrink in ways no gate-level rule would find <d-cite key="duncan-zx-simplify-2020"></d-cite>. PyZX implements this <d-cite key="kissinger-pyzx-2020"></d-cite>, and its methods are especially good at lowering the T-count <d-cite key="kissinger-tcount-zx-2020"></d-cite>.

The hard step is turning the simplified graph back into a circuit, called _extraction_ <d-cite key="backens-extraction-2021"></d-cite>. It can add many CNOTs. That is what happened on the baseline circuit above: the graph was simpler, but the extracted circuit had more than twice the CNOTs. PyZX also has a gentler method, teleport_reduce, that moves phases through the graph and keeps the original circuit's structure; on the same input it lowered the T-count just as far and left the CNOT count unchanged. A review from 2025 surveys the many variants <d-cite key="fischbach-zxreview-2025"></d-cite>.

### Search and learning

The fourth family treats optimisation as a search. Instead of applying whichever rule helps right now, it tries many sequences of rewrites and may accept a worse circuit on the way to a better one, as in row 4 of the figure. Tools of this kind are sometimes called _superoptimisers_, after classical tools that search for the shortest program equivalent to a given one. Quartz generates every small rewrite rule that is valid for a given gate set and verifies each one <d-cite key="xu-quartz-2022"></d-cite>. QUESO synthesises rules whose angles are symbols rather than numbers, and checks them with a randomised test that is right with high probability <d-cite key="xu-queso-2023"></d-cite>. GUOQ, by QUESO's authors, mixes cheap rewrite rules with occasional expensive resynthesis in a search similar to simulated annealing. On its authors' benchmarks it removes 28 percent of two-qubit gates on average, against 18 percent for Quarl, which trains a reinforcement-learning agent to choose the next rule <d-cite key="li-quarl-2024"></d-cite>, and 7 percent for tket <d-cite key="xu-guoq-2025"></d-cite>. QUASAR, presented at PLDI 2026, borrows the _e-graph_ from classical compilers: a data structure that stores many equivalent circuits at once, so the search does not have to commit to one rewrite at a time <d-cite key="yang-quasar-2026"></d-cite>.

These tools spend more compile time, from minutes to hours, for better circuits, and each paper reports results on its own benchmark set, so the percentages are not directly comparable. In production, learning so far shows up in narrow passes. IBM's transpiler service, for example, uses reinforcement learning to synthesise small Clifford, linear (CNOT-only) and permutation blocks and to route <d-cite key="kremer-rltranspile-2024"></d-cite>. Large language models are newer still. In a 2025 experiment on Google's Willow processor, the AlphaEvolve agent, which uses a large language model to write code, evolved the programs that generate the experiment's time-evolution circuits. The authors note that the search scored candidates against a complete, classically computed set of answers, which will not exist for problems beyond classical simulation <d-cite key="zhang-alphaevolve-otoc-2025"></d-cite>.

## Layout, routing and scheduling

On most superconducting chips a two-qubit gate is possible only between neighbouring qubits. The compiler has to choose which physical qubit holds each of the program's qubits (_layout_) and insert SWAP gates to bring distant pairs together (_routing_). A SWAP costs three CNOTs, so routing can multiply a circuit's size.

{% include figure.liquid path="/assets/img/quantum_circuit_optimization/routing-example.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Three panels. 1: four qubits a, b, c, d on a line of physical qubits P0 to P3, and a wanted CX between a and d, which are not connected. 2: a circuit with two SWAPs that move d next to a, then the CX; one CX became seven. 3: a better starting layout a, d, b, c where the CX needs no SWAP." caption="<strong>Routing in miniature.</strong> A CX between two qubits that are not wired together needs SWAPs, each costing three CX. A better starting layout can avoid some of them, but with many gates no single layout suits them all." zoomable=true %}

Choosing layout and SWAPs optimally is NP-complete. Siraichi and co-authors, writing at a classical compiler conference, called it _qubit allocation_, by analogy with register allocation, the classical compiler's job of assigning variables to a processor's few registers <d-cite key="siraichi-qubitallocation-2018"></d-cite>. Production compilers therefore use heuristics. SABRE, the most widely used, looks ahead at the next layer of gates, picks the SWAP that brings them closest together, and runs over the circuit forwards and backwards to choose a good starting layout <d-cite key="li-sabre-2019"></d-cite>. Its successor LightSABRE, written in Rust inside Qiskit, is about 200 times faster than Qiskit's 2020 implementation and uses 18.9 percent fewer SWAPs on average than the original SABRE <d-cite key="zou-lightsabre-2024"></d-cite>.

Exact methods based on SAT and SMT solvers, general-purpose solvers for logical constraints, can find optimal layouts for small circuits <d-cite key="tan-olsq-2020"></d-cite>. Their main use is to measure the heuristics. On benchmark circuits built to have a known optimum, 2020-era tools were on average 1.5 to 12 times deeper than optimal on a small device and 5 to 45 times on a larger one <d-cite key="tan-queko-2021"></d-cite>. A 2025 preprint that counts SWAPs instead finds LightSABRE 63 times above the optimum and tket 330 times <d-cite key="ping-qubikos-2025"></d-cite>. These circuits are built so that the optimum is known, not to resemble real programs, so the gap on real workloads is an open question.

On real devices, a noise-aware compiler chooses the layout and schedule from the latest calibration data, steering gates away from the noisiest qubits <d-cite key="murali-noiseadaptive-2019"></d-cite>. The scheduling stage, which assigns each gate a start time, now also suppresses errors, for example by filling idle periods with dynamical-decoupling pulse sequences, which cancel slow noise on qubits that are waiting <d-cite key="seif-contextaware-2024"></d-cite>.

How much of this you need depends on the hardware. Trapped ions and neutral atoms can connect any pair of qubits, either directly or by moving atoms, so SWAP routing largely disappears and is replaced by scheduling the moves <d-cite key="tan-enola-2025,zhu-qmrsurvey-2025"></d-cite>. The challenge in Part 2 also allowed any pair, so routing played no role there.

## The tools, and how to trust them

### The main tools

Most of the methods above are available in a handful of packages. These are the ones you are most likely to meet:

| tool | what it is | strong at |
| --- | --- | --- |
| Qiskit | IBM's SDK; staged transpiler with a Rust core <d-cite key="javadiabhari-qiskit-2024"></d-cite> | the default for most users; fast routing (LightSABRE) |
| tket | Quantinuum's retargetable compiler <d-cite key="sivarajah-tket-2021"></d-cite> | peephole optimisation, many back ends |
| BQSKit | Berkeley's synthesis-first toolkit <d-cite key="younis-bqskit-2021"></d-cite> | numerical resynthesis, approximate compilation |
| PyZX | ZX-calculus library <d-cite key="kissinger-pyzx-2020"></d-cite> | T-count reduction, experiments with ZX rules |
| VOQC | optimiser with machine-checked proofs <d-cite key="hietala-voqc-2021"></d-cite> | passes proved to preserve the circuit's meaning |
| Quartz, QUESO, GUOQ | research superoptimisers <d-cite key="xu-quartz-2022,xu-queso-2023,xu-guoq-2025"></d-cite> | squeezing a circuit hard, given time |
| MQT (QMAP, QCEC) | Munich toolkit <d-cite key="burgholzer-qcec-2021"></d-cite> | exact mapping, equivalence checking |

The tools can be chained. On the circuits of Part 2 my best gate-level results came from running tket first and Qiskit after it.

### Checking the output

Compilers have bugs, and a buggy optimisation pass returns a circuit that is wrong but looks good. Giallar, a project that verified the passes of Qiskit with an automated prover, covered 44 of 56 passes and found 3 bugs along the way <d-cite key="tao-giallar-2022"></d-cite>. VOQC goes further and proves its passes correct <d-cite key="hietala-voqc-2021"></d-cite>. For everything else there is _equivalence checking_: compare the compiled circuit with the original. QCEC uses the fact that one circuit followed by the inverse of the other must give the identity if the two are equal, and adds simulations on random inputs; its authors report that "in many cases just a single simulation run is sufficient" <d-cite key="burgholzer-qcec-2021"></d-cite>.

I learned to check every output end to end. In the off-the-shelf runs above, one PyZX recipe returned depth 4668, better than every correct result, and that circuit was wrong: the helper qubits did not return cleanly to 0, and some pixels got the wrong sign. During the contest, a tket setting that lets the compiler permute qubits at the end, and Qiskit's default assumption that every qubit starts at 0, also gave me wrong oracles. In each case the tool did what it was told, under an assumption I had not checked.

### Benchmarks

QASMBench <d-cite key="li-qasmbench-2023"></d-cite> and MQT Bench <d-cite key="quetschlich-mqtbench-2023"></d-cite> collect shared benchmark circuits for comparing tools, and Benchpress runs over a thousand tests across seven SDKs <d-cite key="nation-benchpress-2025"></d-cite>. In its 2024 results, measured against Qiskit as the baseline, tket used 1.31 times as many two-qubit gates and 13.3 times the compile time on the geometric mean, and BQSKit 1.26 times the gates and 108 times the time. The authors, who all work for IBM, also write that there is "no clear winner when looking at each test in isolation". Rankings like these move with every release, so treat them as a snapshot.

## The fault-tolerant frontier

Everything so far assumes today's noisy hardware, often called NISQ (noisy intermediate-scale quantum), where each gate adds a little error and the compiler's job is to use fewer gates. A large share of current compiler research targets a different machine: one with _error-corrected_ logical qubits. On such a machine each logical qubit is a patch of hundreds of physical qubits running an error-correcting code, usually the surface code. The cost model changes completely <d-cite key="litinski-gameofsurfacecodes-2019"></d-cite>.

{% include figure.liquid path="/assets/img/quantum_circuit_optimization/ft-stack.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A two-column comparison of a NISQ compiler and a fault-tolerant compiler across six jobs: gates the machine runs, what costs most, rotations, optimise, place and route, and final score. A note under the fault-tolerant column says that magic-state cultivation, proposed in 2024, puts a T state at about the cost of a lattice-surgery CNOT." caption="<strong>Noisy and fault-tolerant compilers compared.</strong> A fault-tolerant compiler does what a NISQ compiler does, but two-qubit Clifford gates become cheap, T gates become the expensive resource, and routing returns at the level of error-corrected patches." zoomable=true %}

### Rotations become T gates

An error-corrected machine runs a small fixed gate set exactly. The usual choice is the Clifford gates (H, S, CNOT) plus the T gate, a rotation by $$\pi/4$$ about $$Z$$. Clifford gates are comparatively cheap to run fault-tolerantly; each T gate needs a magic state, described below. Any other rotation must be approximated by a sequence of these gates. The generic Solovay–Kitaev algorithm does it with a number of gates that grows polylogarithmically in $$1/\varepsilon$$ <d-cite key="dawson-solovaykitaev-2005"></d-cite>. The number-theoretic method gridsynth needs, for a $$Z$$ rotation, typically about $$3\log_2(1/\varepsilon)$$ T gates <d-cite key="ross-gridsynth-2016"></d-cite>, so a rotation accurate to $$10^{-10}$$ costs around a hundred T gates. A 2026 preprint shows that small rotations can be made much cheaper than this angle-independent count suggests <d-cite key="bothe-smallangle-2026"></d-cite>. A circuit that a NISQ compiler considers cheap because its rotations are free can be expensive here.

### Fewer T gates

Because T gates dominate the cost, T-count and T-depth became optimisation targets of their own. A circuit made of CNOT and T gates can be written as a _phase polynomial_, a sum of terms that each say which parity of the qubits receives which phase. Re-synthesising the polynomial gives a polynomial-time method for T-count and T-depth reduction <d-cite key="amy-tdepth-2014"></d-cite>. For circuits of CNOT and T gates, minimising the T-count turns out to be equivalent to decoding a Reed–Muller code <d-cite key="amy-reedmuller-2019"></d-cite>, which led to the TODD optimiser <d-cite key="heyfron-todd-2019"></d-cite>. The ZX-calculus methods above reach the same goal by graph rewriting <d-cite key="kissinger-tcount-zx-2020"></d-cite>, and AlphaTensor-Quantum casts T-count reduction as a tensor decomposition and searches with reinforcement learning <d-cite key="ruiz-alphatensor-2024"></d-cite>.

Fault-tolerant compilers also have to use measurement, which an optimiser that sees only unitaries cannot. A Toffoli gate can be done with four T gates instead of seven if the circuit may measure and correct <d-cite key="jones-toffoli-2013"></d-cite>. A temporary logical AND costs four T gates to compute and none to erase, because the erasure is a measurement followed by a classically controlled Clifford fix-up; this halves the T-count of quantum addition <d-cite key="gidney-halvingaddition-2018"></d-cite>.

### Lattice surgery and magic states

On a surface-code machine, logical operations between patches are done by _lattice surgery_, which merges and splits patches along paths through free space on the chip <d-cite key="litinski-gameofsurfacecodes-2019"></d-cite>. Laying out patches and scheduling those paths is the layout and routing problem again, one level up.

Optimal lattice-surgery compilation is NP-hard <d-cite key="herr-nphard-2017"></d-cite>, so tools use heuristics, such as routing along edge-disjoint paths <d-cite key="beverland-edpc-2022"></d-cite> or SAT solvers for small, heavily reused subroutines <d-cite key="tan-lassynth-2024"></d-cite>, and open end-to-end compilers now exist <d-cite key="watkins-liblsqecc-2024"></d-cite>. There is no agreement on the best overall strategy. One family rewrites the whole computation as a sequence of measurements of multi-qubit Pauli operators, products of X, Y and Z on several qubits, which removes every Clifford gate; another compiles Clifford and T gates directly and keeps more parallelism. A 2026 comparison finds that each wins on different programs <d-cite key="leblond-comparison-2026"></d-cite>.

Each T gate consumes a _magic state_, a special state prepared on the side, and preparing magic states has long been the dominant cost in estimates of what a computation needs. Distillation, which turns many noisy copies into fewer good ones, became cheaper than it was assumed to be <d-cite key="litinski-distillation-2019"></d-cite>. In 2024 Gidney, Shutty and Jones proposed magic-state _cultivation_, which grows a good T state inside a single patch and, in its authors' words, "uses roughly the same number of physical gates as a lattice surgery CNOT gate of equivalent reliability" <d-cite key="gidney-cultivation-2024"></d-cite>. A 2025 experiment on a superconducting processor, reported in a preprint, cut the error of a T state by a factor of 40, keeping 8 percent of attempts and discarding the rest <d-cite key="rosenfeld-cultivationexpt-2025"></d-cite>. If T states become about as cheap as CNOTs, T-count stops being the whole bill. Routing starts to matter as much, and so does timing: cultivation succeeds at random, so a schedule fixed in advance cannot say when a T state will be ready, and part of compilation moves to run time <d-cite key="hofmeyr-puremagic-2025"></d-cite>.

### A resource estimate is the final score

A NISQ compiler's output can be run today. A fault-tolerant compiler's output mostly cannot yet, so its final score is a _resource estimate_: how many physical qubits and how much time the computation would need on a stated hardware model. Estimators such as Microsoft's Azure Resource Estimator <d-cite key="vandam-azurere-2023,beverland-assessing-2022"></d-cite> and Google's Qualtran <d-cite key="harrigan-qualtran-2024"></d-cite> turn logical gate counts into those numbers. The headline estimates have moved fast. Factoring a 2048-bit RSA key was estimated at 20 million noisy qubits for 8 hours in 2021 <d-cite key="gidney-ekera-rsa-2021"></d-cite>. In 2025 the estimate fell to under a million qubits for under a week, under the same hardware assumptions <d-cite key="gidney-rsa-2025"></d-cite>. The author credits approximate arithmetic, denser storage of idle qubits and cultivation. None of the three is a change a gate-level optimiser could make.

Not everything has to be rebuilt for the new costs. A 2025 study found that ordinary NISQ optimisation passes do lower fault-tolerant resource estimates, if chosen per application <d-cite key="forster-ft-era-2025"></d-cite>. The newest survey of fault-tolerant compilers, from September 2026, lists the open problems: optimising across layers of the stack, compiling for codes other than the surface code, designing compilers and decoders (the classical software that turns error-correction measurements into corrections) together, adapting at runtime, and building shared benchmarks <d-cite key="zhu-ftcompilersurvey-2026"></d-cite>.

## What gate-level passes cannot see

The off-the-shelf tools gained only about 6 percent on the challenge circuit, and I think the main reason is what a transpiler is given. It receives a list of gates and must return a list that implements the same unitary. It is not told that some helper qubits always start in 0, that a block of gates computes a particular Boolean function, or that the program only needs the result on some inputs, so it cannot use any of these facts. In the challenge, six helper qubits start at 0, so 63 of every 64 inputs a transpiler must preserve are never used. Structure can also be lost before the transpiler sees it. A 2026 preprint measured this across Qiskit, tket, Cirq and the MQT tools. After a circuit made a round trip through OpenQASM 3, high-level operations such as multi-controlled gates had become plain gates, and asking the next compiler to re-synthesise them had no effect in any of the 360 cases tested. In one pipeline, passing an eight-qubit Grover circuit through OpenQASM 2 raised its two-qubit gate count by 37.2 percent <d-cite key="ye-synthesis-availability-2026"></d-cite>.

A compiler that knows such facts works at a higher level, and these levels have their own tools. High-level synthesis systems, Classiq's among them, take a functional description of a program, together with constraints and goals, and search for a circuit that meets them <d-cite key="goldfriend-scalable-synthesis-2024,vax-qmod-2025"></d-cite>. Silq makes uncomputation part of the language and checks it with types <d-cite key="bichsel-silq-2020"></d-cite>, and Unqomp and its successor Reqomp add uncomputation to circuits automatically, Reqomp within a budget of helper qubits <d-cite key="paradis-unqomp-2021,paradis-reqomp-2024"></d-cite>. Logic synthesis, mentioned above, works on the Boolean function.

{% include figure.liquid path="/assets/img/quantum_circuit_optimization/abstraction-ladder.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A table-like diagram with four rows: Problem, Boolean function, Gate list and Hardware. For each row it names who works there and the depth reached on the logo challenge: 95 from algorithm design, 1886 from logic synthesis (345 with networks shaped by hand), 4665 from my rescheduler starting at 5329, and not scored for hardware mapping. Arrows between the rows say what each step down forgets or adds." caption="<strong>Who can change what.</strong> Each row is a level at which a circuit can be described, and the arrows say what each step down forgets or adds. A gate-level tool cannot use facts that only the Boolean function or the problem shows. The depths are from the challenge in Part 2." zoomable=true %}

In the challenge in [Part 2](/blog/2026/classiq-logo-phase-oracle-depth-95/), off-the-shelf gate-level tools took the baseline circuit from depth 5329 to 5004, and a rescheduler I wrote, which only reorders gates, took it to 4665. Logic synthesis from the logo's truth table reached 1886. Designing the computation by hand, then searching over the designs, reached 95, with gate-level polishing doing the last stretch from 107.

In practice I would still run an off-the-shelf transpiler on any circuit. It is fast and removes routine waste, and its output can be checked against the input. When I needed far more than a few percent, the savings came from changing the description of the problem, and [Part 2](/blog/2026/classiq-logo-phase-oracle-depth-95/) shows how that went for the logo.

## Further reading

Surveys and one tutorial paper, roughly from broad to narrow:

- quantum circuit optimisation, both hardware-independent and hardware-dependent, including machine-learning methods <d-cite key="karuppasamy-review-2025"></d-cite>;
- the compilation process as a whole, compared with classical compilation <d-cite key="cardama-compilation-survey-2025"></d-cite>;
- intermediate representations <d-cite key="cardama-ir-review-2025"></d-cite>;
- qubit mapping and routing across superconducting, trapped-ion and neutral-atom hardware <d-cite key="zhu-qmrsurvey-2025"></d-cite>;
- circuit optimisation with the ZX-calculus <d-cite key="fischbach-zxreview-2025"></d-cite>;
- compilers for fault-tolerant quantum computing <d-cite key="zhu-ftcompilersurvey-2026"></d-cite>;
- a tile-by-tile introduction to computing with the surface code and lattice surgery <d-cite key="litinski-gameofsurfacecodes-2019"></d-cite>.
