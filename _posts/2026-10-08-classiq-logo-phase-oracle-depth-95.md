---
layout: distill
title: "A depth-95 phase oracle for the Classiq logo"
description: My top-five entry to the Classiq Quantum Circuit Challenge, a phase oracle of depth 95 where the organisers' baseline has depth 5329, built from one compute–uncompute sandwich, two small codes, and a search over the codes instead of the circuit. Part 2 of a series on quantum circuit optimization.
tags: quantum-computing circuit-optimization
categories: quantum-computing
giscus_comments: false
date: 2026-10-08 11:00:00 +0800
featured: false
related_posts: true
bibliography: 2026-10-classiq-depth95.bib
og_image: https://nez0b.github.io/assets/img/classiq_depth95/ladder.png
series: quantum-circuit-optimization
series_part: 2
series_previous_url: /blog/2026/quantum-circuit-compilers-field-guide/
series_previous_label: "Part 1: A field guide to quantum circuit compilers"

authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research

toc:
  - name: Compilers and this challenge
    subsections:
      - name: Why not just run a compiler?
      - name: Where this sits in compiler research
  - name: The challenge
  - name: Depth is about packing
  - name: One sandwich for the whole logo
    subsections:
      - name: Compute, flip the sign, uncompute
      - name: From three sandwiches to one
      - name: The finished circuit, layer by layer
  - name: Row and column codes
    subsections:
      - name: What the codes compute
      - name: One distance for two centre rows
      - name: The middle, one formula for four shapes
      - name: The codes themselves
  - name: Contracts and pairing
    subsections:
      - name: Writing down the freedom
      - name: Why pairing is free
  - name: Searching the codes instead of the circuit
    subsections:
      - name: What is being searched
      - name: A depth estimate in nanoseconds
      - name: What the search found
      - name: Is the estimate a good stand-in for depth?
  - name: Polishing, from 107 to 95
    subsections:
      - name: Why it stops at 95
  - name: What did not work
  - name: What carried the result
  - name: Reproduce it

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
  .d95-lede {
    font-size: 1.08rem;
    line-height: 1.75;
  }
  @media (min-width: 1025px) {
    d-article d-contents {
      grid-row: auto / span 6;
    }
  }
  .d95-ideas {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.65rem;
    margin: 1.5rem 0 2rem;
  }
  .d95-ideas > div {
    border: 1px solid var(--global-divider-color);
    border-radius: 0.45rem;
    padding: 0.85rem;
    background: var(--global-card-bg-color);
    font-size: 0.92rem;
    line-height: 1.5;
  }
  .d95-ideas strong {
    color: var(--global-theme-color);
    display: block;
    margin-bottom: 0.25rem;
  }
  .d95-callout {
    border-left: 4px solid var(--global-theme-color);
    background: var(--global-card-bg-color);
    margin: 1.4rem 0;
    padding: 0.9rem 1.1rem;
  }
  .d95-callout > :last-child {
    margin-bottom: 0;
  }
  .d95-downloads li {
    margin-bottom: 0.3rem;
  }
  d-article details {
    margin: 1.2rem 0;
  }
  d-article details > summary {
    cursor: pointer;
    font-weight: 600;
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

<p class="d95-lede">
The Classiq Quantum Circuit Challenge asked for a quantum circuit that puts a minus sign on the black pixels of the Classiq logo, using one- and two-qubit gates on at most 18 qubits, scored by depth. The organisers' baseline circuit has depth 5329. My entry has depth 95 with 272 CNOTs, and it finished in the top five.
</p>

The short version is four design ideas that together closed most of the gap, plus a final round of polishing:

<div class="d95-ideas">
  <div><strong>One sandwich</strong>compute a little about the pixel, flip the sign once, undo the computation</div>
  <div><strong>Two small codes</strong>a column says how far a shape reaches, a row says how far it sits from the centre</div>
  <div><strong>Contracts</strong>list every word each column and row may say, so the two codes never need each other</div>
  <div><strong>Search the codes</strong>an annealer edits code circuits millions of times per second, steered by a cheap depth estimate</div>
</div>

[Part 1](/blog/2026/quantum-circuit-compilers-field-guide/) is a field guide to quantum compilers, but you don't need it to follow this one. I assume you know what a qubit, a CNOT gate and a circuit diagram are, and I explain each compiler or logic-synthesis idea where it comes up. If you want the long form, there is a 19-page write-up and a companion notebook that redraws every data plot below from the shipped data in under a minute; both are linked in [the last section](#reproduce-it).

{% include figure.liquid path="/assets/img/classiq_depth95/method-map.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A tree diagram: depth 95 with 272 cx (baseline 5329) branches into five parts: one sandwich for the whole logo, row and column codes, contracts and pairing, searching the codes instead of the circuit, and polishing the finished circuit, each with a short note on what it does and, for most, the depths it reached." caption="<strong>The method as a map.</strong> Branches 1 to 4 design the circuit; branch 5, polishing, packs a finished circuit more tightly." zoomable=true %}

## Compilers and this challenge

### Why not just run a compiler?

Qiskit and tket are free, and both will shorten a circuit if you ask them to. The obvious first move is to give them the baseline circuit at their strongest setting. I did that on my own reproduction of the baseline, which scores the same 5329 and 3502. Every pair of qubits could share a CNOT, as in the challenge, and I checked each output the same way I checked my entry:

| tool | depth | CNOTs |
| --- | ---: | ---: |
| the baseline itself | 5329 | 3502 |
| Qiskit, optimization level 3 | 5021 | 3355 |
| tket's FullPeepholeOptimise, then Qiskit | 5004 | 3347 |
| PyZX's full_reduce, then extraction | 8950 | 8576 |
| my rescheduler, which only reorders gates | 4665 | 3502 |
| the circuit in this post | 95 | 272 |

The off-the-shelf tools shorten the baseline by about 6 percent, and my own rescheduler by 12.5 percent. PyZX, which rewrites the circuit as a ZX-diagram and turns it back into gates, returned a correct circuit with more than twice the CNOTs. It did cut PyZX's count of T gates by a quarter, and T gates are the cost that matters on the fault-tolerant machines of the next section. A second PyZX recipe returned depth 4668, better than every off-the-shelf row, but that circuit failed the check: its helper qubits did not return cleanly to 0, and some pixels got the wrong sign.

These tools are _transpilers_: a transpiler reads a list of gates and must return a list that implements the same unitary, so the result has to be right on all $$2^{18}$$ basis states. Two facts that matter here are not in the gate list.

- The six helper qubits start at 0 and must end at 0. The circuit only has to be right when they start at 0, so 63 of every 64 inputs that a transpiler must preserve are inputs the challenge never uses. Qiskit assumes by default that every qubit starts at 0. The assumption covers all 18 at once, including the twelve that hold the pixel, and with it my earlier circuits came back wrong on superpositions of pixels, so the Qiskit rows above were run with it switched off.
- The 3502 CNOTs are 18 rectangle tests run one after another, which together compute one bit per pixel. The gate list does not mark where one test ends, and nothing in it names the bit.

A rewrite that needs either fact is out of a transpiler's reach. So the lever in this challenge is not the compiler. It is the _encoding_: which intermediate bits the circuit computes, which qubits they live on, and how the sign is read off them. Here is the whole path from the baseline to 95, one level at a time:

| what changed | depth |
| :-- | --: |
| nothing: the 18 rectangle tests, one after another | 5329 |
| gate-level tools reorder and rewrite the same gates | about 5000, 4665 at best |
| a generic encoding: the pixel function's truth table written as an XOR of products | 1886 |
| logic networks shaped to the logo by hand | 468, then 345 |
| the row and column codes of this post, then a search over the codes | 107 |
| gate-level polishing of that circuit | 95 |

<div class="d95-callout" markdown="1">
**The code design does the work, not the compiler**

Of the 56-fold drop from 5329 to 95, gate-level tools account for a few percent at the start and the last 12 layers at the end. Everything in between came from changing _what_ the circuit computes. A compiler packs the gates it is given; the encoding decides which gates there are to pack. Compilers still matter here, since every candidate circuit goes through a scheduler, but the rest of this post is mostly about designing the codes.
</div>

### Where this sits in compiler research

Depth with free one-qubit gates is a cost model for today's noisy hardware, where every layer of gates adds error. Much of the current research on compilers targets fault-tolerant machines instead, which run error-corrected qubits and count costs differently. There, Clifford gates such as CNOT, H and S are comparatively cheap. The expensive gate is the T gate, because each one consumes a _magic state_ that has to be prepared separately <d-cite key="litinski-game-of-surface-codes-2019"></d-cite>. Compilers for that setting minimise the number of T gates and the number of layers they occupy <d-cite key="amy-tpar-2014,ruiz-alphatensor-quantum-2024"></d-cite>, approximate arbitrary rotations with sequences of Clifford and T gates <d-cite key="ross-selinger-gridsynth-2016"></d-cite>, and lay the computation out on a grid of error-corrected patches <d-cite key="litinski-game-of-surface-codes-2019"></d-cite>. Magic-state cultivation, proposed in 2024, prepares a T state with roughly as many physical gates as a lattice-surgery CNOT of the same reliability <d-cite key="gidney-cultivation-2024"></d-cite>, so the number of T gates is no longer the whole bill.

In this challenge an arbitrary one-qubit gate costs one layer. Under fault-tolerant costs the same gate becomes a sequence of Clifford and T gates that grows longer as the approximation is made tighter <d-cite key="ross-selinger-gridsynth-2016"></d-cite>. [Part 1](/blog/2026/quantum-circuit-compilers-field-guide/#the-fault-tolerant-frontier) surveys the fault-tolerant side in more detail.

## The challenge

Put the column $$x\in\lbrace 0,\dots,63\rbrace$$ of a pixel on six qubits and the row $$y$$ on six more. A _phase oracle_ for the logo, the kind of circuit Grover's search algorithm queries <d-cite key="grover-search-1996"></d-cite>, leaves every basis state as it is except for a sign:

$$
U\,|x\rangle|y\rangle|0\rangle^{\otimes 6} \;=\; (-1)^{\mathrm{logo}(x,y)}\,|x\rangle|y\rangle|0\rangle^{\otimes 6},
$$

where $$\mathrm{logo}(x,y)=1$$ on a black pixel. The six extra qubits are _helpers_. The circuit may use them as scratch space, but they must end in $$\lvert 0\rangle$$ again, so that the oracle behaves the same on any superposition of pixels. That makes 18 qubits in all: $$x$$ on $$q_0$$–$$q_5$$, $$y$$ on $$q_6$$–$$q_{11}$$, and the helpers on $$q_{12}$$–$$q_{17}$$.

Under the challenge rules <d-cite key="classiqchallenge"></d-cite>, the circuit may contain `u3` gates (any one-qubit gate) and `cx` gates (CNOTs) between any two qubits. It is scored by its **depth**, the number of layers of gates, with the `cx` count as a tie-breaker. Two gates can share a layer when they act on different qubits.

The logo is the union of four shapes that the challenge defines exactly: a square ($$2\le x\le 26$$, $$29\le y\le 53$$), a bar ($$26\le x\le 49$$, $$39\le y\le 43$$), a small disc with $$(x-55)^2+(y-41)^2\le 42$$, and a big disc with $$(x-40)^2+(y-19)^2\le 72$$. Together they make 1097 of the 4096 pixels black.

{% include figure.liquid path="/assets/img/classiq_depth95/logo-and-pieces.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="The 64 by 64 Classiq logo drawn as four overlapping shapes: a square, a bar, a small disc and a big disc, with the centre rows 41 and 19 marked." caption="<strong>The target: four overlapping shapes on a 64 × 64 grid.</strong> Each shape is symmetric about one of two rows, row 41 (dashed) or row 19 (dotted), and in every column each shape covers a single run of rows. The codes later in this post are built on these two facts." zoomable=true %}

## Depth is about packing

Depth counts layers, and one layer can hold many gates as long as no two of them touch the same qubit. So two pieces of work on _disjoint_ qubits cost the longer of the two, while two pieces on the _same_ qubits cost the sum:

{% include figure.liquid path="/assets/img/classiq_depth95/packing-rule.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Two timelines. Left: block A takes 3 layers on qubits q0 to q2 while block B takes 2 layers on q3 to q5 in the same layers, so the total is max(3, 2) = 3. Right: A and then B both on q0 to q2, with q3 to q5 idle, so the total is 3 + 2 = 5." caption="<strong>The packing rule.</strong> (a) Work on disjoint qubits shares layers and costs the longer piece. (b) Work on the same qubits has to wait its turn and costs the sum." zoomable=true %}

The baseline circuit covers the logo with 18 rectangles and marks them one at a time. Every rectangle test needs all twelve coordinate qubits, so the rectangles cannot overlap in time, and their depths add up to 5329. I wanted a circuit whose expensive parts live on _different_ qubits, so that a _scheduler_, the compiler step that assigns each gate to a layer, can stack them in the same layers.

## One sandwich for the whole logo

### Compute, flip the sign, uncompute

The oracle has one job: multiply each pixel's state $$\lvert x,y\rangle$$ by $$-1$$ if the pixel is black, and leave it alone if it is white. Deciding black or white is a small calculation, and a quantum circuit can't calculate the way a laptop does. Every gate is reversible, so nothing can be overwritten and forgotten. The intermediate results ("how far does the disc reach in this column?", "how far is this row from row 41?") need somewhere to live, and that place is the helper qubits, which start at $$\lvert 0\rangle$$.

Leaving those results on the helpers after the sign flip would break the oracle. The circuit acts on a superposition of all 4096 pixels at once, and a helper still holding, say, a column's radius would hold a different value for different pixels. It would be entangled with $$x$$ and $$y$$, and the pixels would no longer interfere the way a search algorithm such as Grover's needs them to. The fix goes back to Bennett: compute the result, use it, then run the computation backwards, so the helpers return to $$\lvert 0\rangle$$ and the sign is the only trace left. Almost every oracle that uses helper qubits has this three-part shape, called _compute–uncompute_ <d-cite key="bennett-reversibility-1973"></d-cite>:

| step   | what it does                                                                         |
| :----- | :----------------------------------------------------------------------------------- |
| $$E$$         | compute some useful bits from $$x$$ and $$y$$ onto the helpers (the _encoder_) |
| $$\Phi$$      | flip the sign, depending on those bits (the _middle_)                          |
| $$E^\dagger$$ | run the encoder backwards, which returns the helpers to $$\lvert 0\rangle$$    |

The smallest example is a single AND. A Toffoli writes $$a \wedge b$$ onto a helper, a $$Z$$ on the helper flips the sign when that bit is 1, and a second Toffoli clears the helper again. The net effect is $$\lvert a,b\rangle \mapsto (-1)^{ab}\lvert a,b\rangle$$ with the helper back at 0. That case is just a CZ gate, but the same pattern works for functions far too big for one gate.

The middle only multiplies each basis state by $$\pm 1$$, so running $$E$$ backwards afterwards undoes everything except the sign. I call this shape a _sandwich_. A bonus of the exact undo is that the encoder's multi-controlled gates may use cheaper forms that are right only up to a phase that depends on the basis state. Because the middle is a pure sign flip, those phases pass through it untouched, and the undo, built from the inverse of the very same gates, removes them <d-cite key="maslov-relative-phase-toffoli-2016"></d-cite>.

### From three sandwiches to one

My first circuits used one sandwich per shape (the square and the bar shared one, so three in all) and ran them one after another. Each sandwich needed all twelve coordinate qubits, so, by the packing rule, their depths added up. That design stopped at 231 layers.

The way out was to ask each shape's question differently. "Is $$(x,y)$$ inside the small disc?" needs both coordinates. But it splits into two questions that each need only one: "how far up and down does the disc reach in column $$x$$?" and "how far is row $$y$$ from the disc's centre row?" The pixel is inside when the second number is at most the first. The same split works for all four shapes, because in every column each shape covers one run of rows around row 41 or row 19.

So the design I kept uses **one sandwich for the whole logo** and splits its encoder in two:

- a _column code_ $$E_{\mathrm{col}}$$, which reads only the six $$x$$ qubits and writes on four helpers;
- a _row code_ $$E_{\mathrm{row}}$$, which reads only the six $$y$$ qubits and writes on the other two helpers.

The two codes share no qubit (with one late exception, below), so they run in the same layers, and the encoder costs the longer of the two instead of their sum. An early whole-logo circuit started at 260 layers. I then let an LLM-driven program-evolution loop, ShinkaEvolve <d-cite key="lange-shinkaevolve-2025"></d-cite> (in the spirit of FunSearch <d-cite key="romera-paredes-funsearch-2024"></d-cite>), propose rewrites of the construction, each scored on the full compiled circuit. Together with a few rewrites by hand, that took the circuit to 202 layers, already shorter than 231 even though its codes were crude. Exact rewrites of the same design (reordering gate controls, retiming runs of CNOTs, and two shortcuts in the middle) then brought it to 193 and on to 179.

{% include figure.liquid path="/assets/img/classiq_depth95/serial-vs-shared.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Two timelines drawn to the same scale. Top: three per-shape sandwiches in series, 231 layers. Bottom: one whole-logo sandwich in which the column code and the row code run side by side, 202 layers." caption="<strong>Serial sandwiches against one shared sandwich,</strong> drawn to one time scale. (a) Three per-shape sandwiches each need every coordinate qubit, so they run in series. (b) One sandwich for the whole logo: the column code and the row code sit on disjoint qubits and run side by side, and the middle reads both. Even this early version, with long codes, is shorter: 202 layers against 231." zoomable=true %}

Two pricing rules follow from this shape, and the rest of the method leans on both:

<div class="d95-callout" markdown="1">
**Two pricing rules of the sandwich**

1. Every layer of a code is paid **twice**, once on the way in and once on the way out. A layer of the middle is paid once.
2. The two codes run in parallel, so the circuit is roughly as deep as its **longer side**, the slower of the two codes. Shortening the shorter side buys nothing.
</div>

There are no spare qubits: twelve hold the coordinates and only six are helpers. So each code writes its outputs partly on top of its own input qubits, and only the exact undo at the end puts $$x$$ and $$y$$ back. In the final circuit the column side owns ten qubits ($$x$$ plus $$q_{14}$$–$$q_{17}$$) and the row side owns eight ($$y$$ plus $$q_{12}$$ and $$q_{13}$$). A single gate, a Toffoli controlled by $$q_0$$ and the negated $$q_{11}$$ with target $$q_3$$, reads both sides; it saved six CNOTs late in the campaign.

<div class="d95-callout" markdown="1">
**Side note: why four helpers on one side and two on the other**

A reversible circuit can't forget, and that sets how many helpers each code needs. When a code finishes, its wires must still tell every input apart, or the code could not be run backwards. Whatever the summary lumps together has to be remembered somewhere else on the code's wires.

The two summaries lump very differently. The column summary is coarse. Columns at the same distance left and right of a disc's centre get the same radius, and the columns under the square only need "radius at least 8". In the codes of my depth-138 circuit, which split the wires the same way as the final one, the 64 columns give only 19 different outputs, and up to 7 columns share one. Telling 7 columns apart takes 3 more bits, so the six output bits need at least nine wires. The column code uses ten. The row summary is nearly a relabelling of the row: the distance to the centre row, the side, and two flags almost pin $$y$$ down. The 64 rows give 62 different outputs and no more than two rows share one, so one extra bit is enough. Seven outputs plus that bit is the row side's eight wires.
</div>

### The finished circuit, layer by layer

The final 95-layer circuit is one sandwich, and you can see it in the gates. Layers 1–43 hold 131 of the 272 CNOTs, mostly the two codes, with the column side and the row side in the same layers. Layers 44–52 are mostly the middle and hold only 13 CNOTs. Layers 53–95 hold 128 CNOTs, and 127 of them also appear, with the same control and target, in layers 1–43: they are the codes run backwards.

{% include figure.liquid path="/assets/img/classiq_depth95/sandwich.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Schematic of the final circuit: the column code (26 gadgets) and the row code (22 gadgets) side by side in layers 1 to 43, the middle Phi in layers 44 to 52, and the two undo blocks in layers 53 to 95." caption="<strong>The whole oracle is one sandwich.</strong> Two codes side by side, one short middle Φ that reads both, and the exact undo of each code. The layer ranges are those of the final 95-layer circuit." zoomable=true %}

<div class="l-page" markdown="1">
{% include figure.liquid path="/assets/img/classiq_depth95/occupancy.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 1200px) 1200px, 95vw" alt="A grid of 18 qubits by 95 layers in which every gate is a coloured cell. The left and right blocks are near mirror images and the middle block is short." caption="<strong>The final circuit, qubit by qubit and layer by layer.</strong> A coloured cell is a gate on that qubit in that layer. The left and right blocks are near mirror images (the codes and their undo); the middle block is short. The helper q16 is busy in all nine middle layers, which is what stops the circuit at 95." zoomable=true %}
</div>

Depth is set by a few crowded wires. The helper $$q_{16}$$ is busy in 81 of the 95 layers, while $$q_0$$ and $$q_7$$ are busy in only 18. That is why the search later in this post targets the column code and the time its helpers stay busy.

## Row and column codes

### What the codes compute

The sandwich only helps if both codes are short, so the codes should carry as little information as the middle needs. In every column of the logo, the black pixels of each shape form one unbroken run of rows, centred on row 41 or on row 19. So a column can be summarised by _how far from the centre row it still reaches_, a row by _how far it sits from the centre row_, and a pixel is black when the second number is within the first.

| code        | bits                                     | meaning                                                                         |
| :---------- | :--------------------------------------- | :------------------------------------------------------------------------------ |
| column code | $$\mathsf{R}=\mathsf{R}_0\ldots\mathsf{R}_3$$ | a 4-bit _radius_: how far from the centre row this column belongs to a disc |
|             | $$\mathsf{fam}$$                        | "this is a bar column"                                                          |
|             | $$\mathsf{L}$$                          | "this is a disc column"                                                         |
| row code    | $$\mathsf{d}=\mathsf{d}_0\ldots\mathsf{d}_3$$ | a 4-bit _distance_ from this row to its centre row (41 or 19)               |
|             | $$\mathsf{s}$$                          | 1 if the row is above its centre row                                            |
|             | $$\mathsf{f}$$                          | "this row crosses the square" ($$29\le y\le 53$$)                               |
|             | $$\mathsf{b}$$                          | "this row crosses the bar" ($$39\le y\le 43$$)                                  |

These meanings only need to hold where the middle reads them. Elsewhere a code may write other values, as long as the picture stays right; the examples below show some.

### One distance for two centre rows

The row code has to measure distances to two different centre rows with the same four bits. It does this with a subtraction. It computes $$\text{centre}-y$$ in four-bit binary and keeps the sign of the result as $$\mathsf{s}$$. On or below the centre row the result is the distance. Above it the result is negative, and the code is left holding $$y-\text{centre}-1$$, one less than the distance (the negative result's four bits, inverted). Instead of spending gates to fix this, the middle tests $$\mathsf{d}+\mathsf{s}\le\mathsf{R}$$, which is the true distance on both sides.

{% include figure.liquid path="/assets/img/classiq_depth95/distance-code.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Two panels, one above the other. Top: rows of the big disc measure from row 19, rows of the other shapes from row 41. Bottom: why flipping bits of y does not work for centres 19 and 41, and how a subtraction with a sign bit does." caption="<strong>One 4-bit distance for both centre rows.</strong> (a) Rows of the big disc measure from row 19; rows of the square, bar and small disc from row 41. (b) A simpler trick, flipping bits of y, would need the two centre rows to differ in every bit, and 19 and 41 do not. So the code subtracts, keeps the sign bit s, and the middle compares d + s with the radius." zoomable=true %}

### The middle, one formula for four shapes

The middle reads the thirteen code bits and flips the sign when

$$
\Phi \;=\;
\underbrace{\mathsf{fam}\,\mathsf{b}\,\mathsf{f}}_{\text{bar}}
\;\oplus\; \underbrace{(\mathsf{f}\oplus\mathsf{fam})\,\mathsf{L}\,\mathsf{X}\,\mathsf{C}}_{\text{discs}}
\;\oplus\; \underbrace{\mathsf{L}\,\mathsf{R}_3}_{\text{correction}}
\;\oplus\; \underbrace{\mathsf{f}\,\mathsf{R}_3}_{\text{square}}
$$

is 1, where $$\mathsf{X}=\mathsf{d}_3\oplus\overline{\mathsf{R}_3}$$ and $$\mathsf{C}=\mathbf{1}\left[(\mathsf{d}\bmod 8)+\mathsf{s}\le(\mathsf{R}\bmod 8)\right]\oplus\mathsf{R}_3$$. Products are ANDs of bits, $$\oplus$$ is addition mod 2 (XOR), $$\overline{\mathsf{R}_3}$$ is the NOT of $$\mathsf{R}_3$$, and $$\mathbf{1}[\cdot]$$ is 1 when the condition inside holds and 0 otherwise.

Three of the four terms need no arithmetic. The bar is a column flag times two row flags, and the square is a row flag times the top radius bit. The correction term covers the five columns $$x=38,\dots,42$$, where the big disc's half-height is 8 and so needs the fourth radius bit. The disc term holds the one real comparison: on a disc column with radius below 8, the product $$\mathsf{L}\,\mathsf{X}\,\mathsf{C}$$ is exactly $$\mathbf{1}[\mathsf{d}+\mathsf{s}\le\mathsf{R}]$$. That comparison is why the middle contains a small adder-like circuit, a _ripple-carry comparator_ <d-cite key="cuccaro-ripple-carry-2004"></d-cite>.

The remaining factor, $$\mathsf{f}\oplus\mathsf{fam}$$, picks the right centre row. The big disc's columns are also bar columns ($$\mathsf{fam}=1$$), and there the factor keeps the test to rows outside the square's band of rows 29–53, which hold all of the big disc's rows (11–27, measured from row 19). On the small disc's columns ($$\mathsf{fam}=0$$) it keeps the test to rows inside the band, where the small disc's rows (35–47) measure from row 41.

After the comparator, the middle ends in a _phase block_ that flips the sign of the disc term. A product of four bits can be turned into a sign by 15 small rotations, one on the XOR of each nonempty subset of the four bits, and that is how the block does it. Its layers on the helper $$q_{16}$$ set the final depth, as [Why it stops at 95](#why-it-stops-at-95) shows.

The terms are joined by XOR rather than OR because a sign flip by an XOR of products splits into one sign flip per product, $$(-1)^{a\oplus b}=(-1)^a(-1)^b$$, so the middle can apply the terms one at a time. This is the standard way to turn an XOR of products into a phase oracle <d-cite key="schmitt-boolean-to-quantum-2021"></d-cite>. The price is that a pixel covered by two terms cancels, so the formula is written to make the cancellations land exactly on the white pixels.

<div class="l-page" markdown="1">
{% include figure.liquid path="/assets/img/classiq_depth95/middle-terms.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 1200px) 1200px, 95vw" alt="Five 64 by 64 panels: the bar term, the disc term, the correction term, the square term, and their XOR, which is exactly the logo." caption="<strong>The four terms of the middle's formula,</strong> evaluated on the real codes of one of my circuits, and their XOR. Each black region is where that term equals 1. Overlaps outside the logo cancel, and the XOR (right) is the logo with no wrong pixel." zoomable=true %}
</div>

Here are three pixels traced through the codes of my depth-138 circuit, whose column and row codes were found by two separate searches:

| pixel $$(x,y)$$       | $$\mathsf{R}$$ | $$\mathsf{fam}$$ | $$\mathsf{L}$$ | $$\mathsf{d}$$ | $$\mathsf{s}$$ | $$\mathsf{f}$$ | colour                                          |
| :-------------------- | -------------: | ---------------: | -------------: | -------------: | -------------: | -------------: | :---------------------------------------------- |
| (55, 44), small disc  | 6              | 0                | 1              | 2              | 1              | 1              | $$\mathsf{d}+\mathsf{s}=3\le 6$$: **black**     |
| (55, 48), above it    | 6              | 0                | 1              | 9              | 0              | 1              | $$\mathsf{d}+\mathsf{s}=9>6$$: **white**        |
| (10, 30), square      | 13             | 0                | 0              | 11             | 0              | 1              | $$\mathsf{f}\,\mathsf{R}_3=1$$: **black**       |

At $$x=55$$ the small disc reaches 6 rows from row 41, and $$\mathsf{R}=6$$ says so. Row 44 is 3 above row 41: the code stores $$\mathsf{d}=2$$, $$\mathsf{s}=1$$, and $$2+1\le 6$$. Row 48 is 7 above row 41. There the code stores $$\mathsf{d}=9$$ and $$\mathsf{s}=0$$, neither the distance nor the right side, but that is just as good: any $$\mathsf{d}+\mathsf{s}$$ above 6 gives the right answer, white. In the square column $$x=10$$ the radius is 13, but the middle only reads its top bit $$\mathsf{R}_3=1$$, and $$\mathsf{f}\,\mathsf{R}_3$$ marks the pixel black.

### The codes themselves

Where a disc needs the number, the code is exact: the radius matches the disc's true half-height in every disc column, and the distance matches the true distance in every row a disc covers. Elsewhere the code wanders. Under the square the radius ranges from 10 to 15 (any value from 8 to 15 would do), because only $$\mathsf{R}_3$$ is read. Far from every disc the distance only has to be large enough. The search found these codes, and it used that slack to make them shorter.

<div class="l-page" markdown="1">
{% include figure.liquid path="/assets/img/classiq_depth95/codes.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 1200px) 1200px, 95vw" alt="Two bar charts. Left: the radius written for each of the 64 columns, with ticks at each disc's true half-height. Right: d plus s written for each of the 64 rows, with a line at the true distance to row 19 or row 41." caption="<strong>The two codes of my depth-138 circuit.</strong> Left: the radius R written for each column; black ticks mark the disc's true half-height, which R matches on every disc column. Right: d + s written for each row (the purple cap is s); the black line is the true distance to row 19 or row 41. The codes agree with the geometry where a disc reads them and are free to differ elsewhere." zoomable=true %}
</div>

## Contracts and pairing

### Writing down the freedom

The square columns showed that a code has choices: wherever the middle does not read a bit, any value works. I made this exact. For every column $$x$$ I list the six-bit _words_ (values of the six output bits, read together) that the column code may write there, and for every row $$y$$ the words the row code may write. (The row code has one spare output bit that the middle never reads, so row words have eight bits.) A word is allowed if it still draws the right picture. Chip-design tools call such free outputs _don't-cares_, and exact synthesis of reversible circuits exploits them in the same way <d-cite key="grosse-exact-dontcares-2008"></d-cite>.

A column word is only right or wrong _together with_ a row word, so the two lists depend on each other. I settle that by starting from one pair of codes known to be correct, those of my depth-138 circuit, and pruning until nothing changes:

```text
start   a column may write any word that is right against the pair's
        row code, and a row any word right against its column code
repeat  drop a column word if it draws a wrong pixel against SOME
        allowed word of SOME row
        drop a row word if it draws a wrong pixel against SOME
        allowed word of SOME column
until   nothing is dropped
```

What survives is the **contract**, and it still contains the starting pair. For the middle above it allows 323 column words, between 1 and 9 per column, and 2109 row words, between 4 and 60 per row. The column side splits into four groups:

| columns                                 | allowed words | what is fixed                                                  |
| :-------------------------------------- | ------------: | :------------------------------------------------------------- |
| 0–1 and 62–63 (empty edges)             | 8             | $$\mathsf{R}_3=\mathsf{fam}=\mathsf{L}=0$$                     |
| 2–26 (square)                           | 8             | $$\mathsf{R}_3=1$$, $$\mathsf{fam}=\mathsf{L}=0$$              |
| 27–31, 49 and 61 (bar only, disc edges) | 9             | only $$\mathsf{R}_3=0$$                                        |
| 32–48 and 50–60 (discs)                 | 1             | all six bits                                                   |

The groups follow the picture: under the discs the radius must be exact, and under the square only $$\mathsf{R}_3$$ matters. On the row side, rows that cross no shape may use up to 60 different words.

### Why pairing is free

Because the pruning ran until nothing changed, every surviving column word is right against _every_ surviving row word. So:

<div class="d95-callout" markdown="1">
**Pairing.** Any column code that writes allowed words on all 64 columns works with any row code that writes allowed words on all 64 rows. With $$N$$ column codes and $$M$$ row codes, $$N+M$$ independent searches give $$N\times M$$ correct circuits.
</div>

This splits one hard joint search into two smaller ones that can run on different machines and never talk to each other. Checking a candidate code also becomes cheap. A column code is correct if it writes an allowed word on each of the 64 columns, which is 64 table lookups instead of 4096 pixels.

{% include figure.liquid path="/assets/img/classiq_depth95/contract-and-pairing.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="Left: allowed column and row words pruned in a loop until nothing changes. Right: any legal column code connects to any legal row code." caption="<strong>The contract, and why it makes pairing free.</strong> Left: allowed words are pruned until nothing changes, and what remains is the contract. Right: because each column word is right against every allowed row word, any legal column code pairs with any legal row code." zoomable=true %}

Here is one real pairing run: 6 annealed column codes against 4 annealed row codes, all 24 combinations built and compiled. Every pair draws the logo with no wrong pixel. Changing the column code moves the depth by up to 2 layers. Three of the four row codes give identical depths, and the fourth adds one layer everywhere. Because the column side is the longer side, a better row code buys nothing until the column code improves, which is the second pricing rule. From here on, I scored each side alone by compiling a _half circuit_: one code, the middle, and that code's undo.

{% include figure.liquid path="/assets/img/classiq_depth95/pairing.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A 4 by 6 heat map of compiled depths, one cell per pairing of a row code with a column code; the values vary mostly from column to column." caption="<strong>Twenty-four paired circuits, all correct.</strong> Each cell is the compiled depth of one column code (columns) paired with one row code (rows) from a single pairing run. Most of the variation follows the column code." zoomable=true %}

## Searching the codes instead of the circuit

Up to 179 layers I designed the codes by hand, with ShinkaEvolve proposing rewrites. Every step below 179 used codes found by automatic search.

### What is being searched

A column code is a small reversible circuit on ten qubits: the six $$x$$ qubits and four helpers that start at 0. It is built from NOT gates with zero to three controls (`x`, `cx`, `ccx` and `c3x`, any control of which may be negated), plus a choice of which qubit holds each of the six outputs and whether it is stored inverted. The code is correct if, on every column, it writes a word the contract allows.

I search this space with _simulated annealing_ <d-cite key="kirkpatrick-annealing-1983"></d-cite>, a random walk that proposes a small change, always accepts it if the result is better, and sometimes accepts it if it is worse. The chance of accepting a worse change shrinks as a "temperature" is lowered, which lets the walk climb out of dead ends early on and settle down later. My program, `colsa.c`, is about 300 lines of C. Its moves change a control, flip a control's polarity, change a target or a gate type, insert or delete a gate, swap two neighbouring gates, move a gate elsewhere in the list, move an output to another qubit, or flip whether an output is stored inverted. The idea is close to _stochastic superoptimisation_ <d-cite key="schkufza-stoke-2013"></d-cite>, which searches machine code the same way, scoring each candidate on test cases plus a cheap cost estimate; annealing has also been used to synthesise quantum circuits directly <d-cite key="paradis-synthetiq-2024"></d-cite>.

Each candidate is checked on all 64 columns at once. In the annealer, each qubit is stored as one 64-bit integer whose bit $$x$$ is that qubit's value on column $$x$$, so a gate costs one AND per control and one XOR, over all 64 columns together:

```c
for (int k = 0; k < s->n; k++) {                 /* run the gate list     */
    const Gate *g = &s->g[k];
    uint64_t m = ~0ULL;                          /* columns where all     */
    for (int j = 0; j < ncontrols(g->type); j++) /* controls hold ...     */
        m &= g->p[j] ? ~v[g->c[j]] : v[g->c[j]];
    v[g->t] ^= m;                                /* ... flip their target */
}
/* then read the six output wires and count columns whose word is not allowed */
```

### A depth estimate in nanoseconds

Compiling a candidate into `u3` and `cx` gates and scheduling it takes seconds. My compile chain runs a tket <d-cite key="sivarajah-tket-2021"></d-cite> _peephole_ pass (local rewrites of short runs of gates), the Qiskit transpiler <d-cite key="javadi-abhari-qiskit-2024"></d-cite> at its highest optimisation level, and then my own scheduler, which reorders gates that _commute_, that is, gates whose order does not change what the circuit computes. The annealer needs an answer in nanoseconds, so it uses an estimate, $$\mathrm{est}$$: the as-soon-as-possible depth of the gate list (each gate placed in the first layer where its qubits are free) with fixed prices of 1 layer per `cx`, 7 per `ccx` and 13 per `c3x`, since a multi-controlled gate becomes a _gadget_, a block of several layers of `u3` and `cx` gates, once compiled <d-cite key="barenco-elementary-gates-1995"></d-cite>. A refined version also charges for how late each output becomes ready. The middle reads some outputs early, and an output it waits for delays the middle and, through the undo, the end of the circuit as well.

The walk minimises the energy

$$
\text{energy} \;=\; 20\cdot\text{wrong} \;+\; 3\cdot\max(0,\ \mathrm{est}-\text{cap}) \;+\; 0.01\cdot\mathrm{est} \;+\; 0.001\cdot\text{gates},
$$

where "wrong" counts columns whose word breaks the contract. Each time the walk finds a correct code with $$\mathrm{est}\le\text{cap}$$, it prints the code and lowers the cap to $$\mathrm{est}-1$$, so it keeps pushing for a shallower code. Simplified from `colsa.c`, the main loop is:

```c
double T = T0 * pow(T1 / T0, (double)it / iters);       /* cool down slowly   */
State t = s;  mutate(&t);                               /* propose a change   */
double e2 = energy(&t, cap, &w2, &est2);
if (e2 <= e || runif() < exp((e - e2) / T)) {           /* Metropolis rule    */
    s = t;  e = e2;
    if (w2 == 0 && est2 <= cap) { print_state(&s, est2, it); cap = est2 - 1; }
}
```

One core checks between two and three million candidates per second this way.

### What the search found

The first large run used 24 _chains_, independent annealing walks, of 300 million steps each: 12 from random gate lists and 12 seeded with my best hand-made column code. Nine of the twelve random chains never reached a correct code, and the three that did got there late. But those three found codes with estimates of 69, 72 and 76. The seeded chains all repaired the hand-made code (which, against this run's contract, was wrong on 5 columns) and then stalled between 78 and 96, close to the code they started from.

The code with estimate 72 became a circuit of depth 147, thirty-two layers below the hand-made 179 in one step. Annealing the row side the same way and pairing gave 138. From there, rounds of _anneal, compile, keep the best_ on each side, a SAT-scheduled phase block, and the refined estimate brought the circuit to 107.

<div class="l-page" markdown="1">
{% include figure.liquid path="/assets/img/classiq_depth95/annealer.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 1200px) 1200px, 95vw" alt="Left: the number of wrong columns against annealing step for twelve random-start chains on a log scale. Right: the best estimate of any correct code found by each chain." caption="<strong>The first annealing run.</strong> Left: columns still wrong against annealing step for the 12 random-start chains (log scale). Green chains reached a correct code; the dashed orange trace is a 20-million-step chain run live in the companion notebook. Right: the best estimate of any correct code per chain." zoomable=true %}
</div>

### Is the estimate a good stand-in for depth?

The annealer never sees the real depth, so the method works only if a lower $$\mathrm{est}$$ means a shallower circuit. There are three reasons to expect it.

- **The codes are most of the circuit.** In the 106-layer circuit, removing the middle's phase block left 93 layers. In the 95-layer circuit, everything except that block fits in 86 layers.
- **The prices roughly follow the compiler.** A multi-controlled gate costs several layers once compiled, and the estimate charges it that way.
- **Codes are paid twice.** By the first pricing rule, a layer saved in the longer code saves two layers of the circuit, so the estimate measures the part of the circuit that counts double.

The plot below tests it. On the left, codes from my archive of 296,770 column codes are grouped by estimate and compiled. Lower estimate means shallower circuits: the median compiled half circuit climbs from 111 to 131 layers as the estimate goes from 57 to 64. Over 13,830 compiled half circuits, the refined estimate ranks codes in nearly the right order: its rank correlation with compiled depth, which would be 1 for a perfect ordering and 0 for a random one, is 0.89 (its prices were fitted on those same circuits), against 0.87 for plain as-soon-as-possible depth.

The right panel, from a benchmark taken earlier in the search, shows the limit: sixty codes with estimates of 59 or 60 compiled to anything from 127 to 148 layers. Near the best codes the estimate does worse still. The best codes all share an estimate of 57, so it cannot order them at all, and even timing prices refitted on those codes rank unseen ones at only 0.27. In the first run, the code with estimate 69 became a circuit of 167 layers, while the code with estimate 72 gave the 147. The spread comes from the last step of compilation, the scheduler that reorders gates that commute. Depth after the earlier compile steps ranks the final depth at only 0.35 to 0.36, and no cheap formula I tried predicted the scheduler.

<div class="l-page" markdown="1">
{% include figure.liquid path="/assets/img/classiq_depth95/proxy.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 1200px) 1200px, 95vw" alt="Left: median compiled depth of archive codes rising with the estimate. Right: a histogram of compiled depths of 60 codes estimated at 59 or 60, spread over 21 layers." caption="<strong>How well the estimate predicts compiled depth.</strong> Left: archive codes grouped by estimate and compiled into half circuits; lower estimates give shallower circuits on average. Right: in an earlier benchmark, 60 codes with estimates of 59 or 60 spread over 21 layers once compiled." zoomable=true %}
</div>

So the search ran as a funnel. The annealer explores with the estimate, every correct code it prints is compiled, and only compiled depth decides what is kept and what seeds the next round. STOKE, the superoptimiser mentioned above, works the same way: it searches with cheap test cases and keeps only programs that a formal check validates <d-cite key="schkufza-stoke-2013"></d-cite>, and recent quantum circuit optimisers likewise mix cheap and expensive moves in one randomised search <d-cite key="xu-guoq-2025"></d-cite>.

{% include figure.liquid path="/assets/img/classiq_depth95/search-funnel.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A three-stage loop: anneal with the estimate, compile every correct code, keep the best by compiled depth, and feed the best back into the next round of annealing." caption="<strong>The search loop.</strong> The best codes of each round seed the next round, and the best column and row codes are paired." zoomable=true %}

## Polishing, from 107 to 95

At 107 layers the code search stopped improving: every search I ran came back with the same best column code. The last twelve layers came from tools that change the circuit _without changing what it computes_. Most of them either reorder gates that commute <d-cite key="itoko-scheduling-2020"></d-cite>, or rebuild a small block of the circuit with an exact solver <d-cite key="peham-depth-optimal-clifford-sat-2023"></d-cite>: a SAT solver, which decides whether a set of yes/no constraints can all hold at once, or CP-SAT, its cousin that also optimises.

| depth / cx | what bought it                                                                                                                    |
| ---------: | :-------------------------------------------------------------------------------------------------------------------------------- |
| 107 / 279  | one gate that reads both sides (the Toffoli from $$q_0$$ and the negated $$q_{11}$$ onto $$q_3$$): six fewer CNOTs, same depth     |
| 106 / 279  | the phase block has 238 equivalent schedules; one of them saves a layer                                                           |
| 98 / 269   | a cheaper three-control gadget, the full tket pass, and an exact CP-SAT <d-cite key="perron-cpsat-software-2024"></d-cite> reschedule |
| 97 / 267   | how each multi-control gadget is compiled, chosen for pairs of neighbouring gadgets at once                                       |
| 96 / 273   | one-qubit gates moved out of runs of gates they commute with                                                                      |
| 95 / 279   | the phase block rebuilt by a SAT solver <d-cite key="biere-cadical-2-2024"></d-cite> inside the rest of the circuit               |
| 95 / 272   | three more small regions rebuilt with fewer CNOTs; the last, the phase block merged with the sign gates, now has 19, the fewest that region can have |

The ladder below puts these steps at the end of the whole history. The largest single drops came from changing what I searched over: one sandwich instead of three, then codes instead of circuits. No polishing step gained more than eight layers.

<div class="l-page" markdown="1">
{% include figure.liquid path="/assets/img/classiq_depth95/ladder.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 1200px) 1200px, 95vw" alt="Depth of fourteen stored circuits in the order they were found, falling from 231 to 95, coloured by whether they were designed by hand, found by searching codes, or polished." caption="<strong>My circuits from depth 231 to 95,</strong> each point a stored circuit re-measured by the companion notebook. Grey: codes designed by hand. Orange: codes found by search. Blue: a finished circuit packed more tightly. The baseline, 5329, is far off the top of the plot." zoomable=true %}
</div>

### Why it stops at 95

As the occupancy plot showed, the helper $$q_{16}$$ is busy in every one of the nine middle layers. Every longest chain of gates in the circuit runs through the phase block on $$q_{16}$$. If that block takes $$K$$ layers on $$q_{16}$$, the whole circuit needs exactly $$86+K$$ layers, for every $$K$$ from 0 to 9. And the block cannot take fewer than 9 layers on $$q_{16}$$: the solver proves 8 impossible even when the block's other qubits get all the time they want. So $$86+9=95$$. On this circuit, then, 94 needs either a phase block that spends fewer layers on $$q_{16}$$, which the solver rules out for this block on its own qubits, or new codes that let the rest fit in fewer than 86 layers. A screen of all 1,738 circuits I had stored, with the same test, found 534 that fit 95 and none that fit 94.

## What did not work

Besides polishing, I spent a long time trying to make the codes themselves shorter than those of the 106-layer circuit. Twelve routes failed, three for each of four reasons. Some ran out of _helper qubits_: a clean spare qubit would have helped, and with six helpers neither side could give one up. Some made one side cheaper only by making the other side's contract tighter, and the column side, which sets the depth, always paid. Some searches spent their whole budget and returned nothing. And some rested on a number measured in one setting and reused in another. For example, the phase block's 238 equivalent schedules had been ranked once, against an older 113-layer circuit; ranking them again inside the 107-layer circuit was the free layer in the polishing table.

<details markdown="1">
<summary>The closed routes, sorted by why they failed</summary>

{% include figure.liquid path="/assets/img/classiq_depth95/dead-ends.png" class="img-fluid rounded z-depth-1" sizes="(min-width: 930px) 930px, 95vw" alt="A tree of twelve closed routes in four groups: helper-starved, contract-starved, search-failed and measurement error, each with a one-line reason." caption="<strong>Closed routes, sorted by why they failed.</strong> Three of the four reasons are about resources (helper qubits, contract room, search budget), and the fourth is about stale measurements." zoomable=true %}

</details>

## What carried the result

Five things carried the result:

1. **Pick a shape that packs.** One sandwich with the two codes on disjoint qubits costs the longer code, where per-shape sandwiches cost the sum of all of them.
2. **Describe the picture with small codes.** A radius per column and a distance per row turn four shapes into one comparison and a few products of flags.
3. **Write down the freedom.** The contract lists every word each column and each row may use. It makes the two codes independent, so pairing costs nothing, and it makes checking a candidate a table lookup.
4. **Search with a cheap compass, decide with the real compiler.** The estimate steered billions of annealing steps; only compiled depth decided what I kept.
5. **Polish last.** Exact rescheduling and SAT re-synthesis of small blocks took the final twelve layers without searching the codes again.

## Reproduce it

Everything needed to check the result and rerun the plots is here:

<ul class="d95-downloads">
  <li><a href="/assets/pdf/classiq-depth95/logo-oracle-95-272.qasm">logo-oracle-95-272.qasm</a>: the final circuit, OpenQASM 2 <d-cite key="cross-openqasm-2017"></d-cite> with <code>u3</code> and <code>cx</code> gates on 18 qubits (depth 95, 272 <code>cx</code>).</li>
  <li><a href="/assets/pdf/classiq-depth95/logo-oracle-depth-95-explainer.pdf">The write-up (PDF, 19 pages)</a>: the same method in more detail, with the worked numbers.</li>
  <li><a href="/assets/pdf/classiq-depth95/depth95-explainer.zip">The companion bundle (zip)</a>: a Jupyter notebook, its one helper script, and every circuit, code, contract and search log it reads, including the annealer's C source.</li>
</ul>

To run the notebook, unzip the bundle, install `numpy`, `matplotlib` and `jupyter` (Python 3.9 or newer), and run all cells from inside the folder. It needs no Classiq account and no network, and it takes under a minute. A C compiler is optional: with one, the notebook compiles the annealer and runs a 20-million-step search live; without one, it plots a stored run of the same search.

Besides my own annealer, the work relied on open tools. Code circuits were compiled with tket <d-cite key="sivarajah-tket-2021"></d-cite> and the Qiskit transpiler <d-cite key="javadi-abhari-qiskit-2024"></d-cite>. Exact rescheduling used the CP-SAT solver of Google OR-Tools <d-cite key="perron-cpsat-software-2024"></d-cite>. The phase block was re-synthesised as a SAT problem, encoded with PySAT <d-cite key="ignatiev-pysat-2018"></d-cite> and solved with CaDiCaL <d-cite key="biere-cadical-2-2024"></d-cite> and Kissat <d-cite key="biere-kissat-satcomp-2024"></d-cite>. The first whole-logo circuits were evolved with ShinkaEvolve <d-cite key="lange-shinkaevolve-2025"></d-cite>.

Thanks to the Classiq team for a well-posed challenge with an exact scoring rule, and for encouraging participants to publish their solutions.
