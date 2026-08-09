---
layout: distill
title: Belief Propagation
description: Exact inference on trees, principled approximations on loopy graphs, and a common language for optimization, coding, learning, and tensor contraction
permalink: /projects/belief-propagation/
tags: graphical-models belief-propagation message-passing combinatorial-optimization
img: assets/img/belief-propagation/cover.png
importance: 1
category: work
show_on_projects: true
series: belief-propagation
status: draft
giscus_comments: false
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: belief-propagation.bib
toc:
  - name: One algorithm, several languages
  - name: Reading map
  - name: The recurring questions
  - name: Conventions and reproducibility
---

<div class="bp-series-nav">
  <div class="bp-draft"><strong>Draft series</strong> · four chapters drafted; derivations, figures and citations under review</div>
  <div class="bp-series-kicker">Belief propagation · exact trees, approximate loops, and messages as computation</div>
</div>

{% include figure.liquid path="assets/img/belief-propagation/cover.png" alt="A factor graph with circular variable nodes and square factor nodes carrying messages in both directions toward a highlighted root marginal" caption="Belief propagation reorganizes a global marginalization into local messages. On a tree this is exact dynamic programming; on a loopy graph the same equations become a fixed-point approximation." %}

## One algorithm, several languages

Belief propagation is easy to recognize and surprisingly hard to place. In a probabilistic graphical model it is the **sum–product algorithm**. On a tree it is ordinary dynamic programming. In statistical physics it is the replica-symmetric **cavity method**. In coding theory it is iterative decoding. In combinatorial optimization its zero-temperature limit becomes **min-sum** or **max-product**. Graph neural networks borrow the same local aggregation pattern, while tensor-network algorithms use closely related messages to approximate an environment.

These are not merely metaphors. They share a concrete computational move: cut an edge, summarize everything on one side as a function of the boundary variable, and pass that summary across the cut. What changes from field to field is the algebra carried by the message, the assumptions under which branches become independent, and the meaning attached to a fixed point.

This series develops that common spine from the exact case outward. The goal is not to memorize an update rule. It is to understand where the rule comes from, what it computes, why it is exact on trees, what the Bethe free energy adds, and which claims stop being theorems when the graph contains loops.

## Reading map

{% include series_chapter_cards.liquid series="belief-propagation" %}

### Planned chapters

- **Error-correcting codes:** parity-check factor graphs, log-likelihood ratios, density evolution, trapping sets, and the gap between tree-like analysis and finite codes.
- **Quantum circuits and tensor networks:** environments, gauges, contraction on loopy networks, and belief-propagation approximations to PEPS and circuit amplitudes. This topic will likely require several chapters.

The roadmap names intended directions; it does not present unfinished chapters as published results.

## The recurring questions

Every chapter will return to five questions.

1. **What global quantity is being computed?** A marginal, a partition function, a minimum-energy assignment, a codeword posterior, or an approximate tensor environment are different targets.
2. **What does one message summarize?** A message is always conditional on an edge being removed, but it may store probabilities, log-likelihood ratios, energies, warnings, tensors, or learned features.
3. **Where does factorization enter?** On a tree, pinning the boundary disconnects branches exactly. On a loopy graph, using the same product is an approximation whose quality depends on correlation decay and loop structure.
4. **What does convergence mean?** A numerical iteration can converge to an inaccurate belief, fail to converge despite an accurate Bethe approximation, or settle at a non-global stationary point.
5. **What is independently checkable?** Small instances can be enumerated exactly; tree identities can be tested to machine precision; thresholds and phase diagrams require their ensemble and stability assumptions to be stated explicitly.

That last question matters for a public technical series. “BP works” is never a complete claim. The graph family, objective, update schedule, initialization, and comparison target all belong in the sentence.

## Conventions and reproducibility

The first two chapters use finite-alphabet factor graphs. Variable nodes are circles, factor nodes are squares, and a directed message is named by its sender and receiver. Free entropy means $\log Z$ (or its density); free energy differs by the conventional factor $-1/\beta$. Whenever a zero-temperature limit is taken, additive message gauges are fixed explicitly.

The derivations are adapted from standard factor-graph, variational-inference, and statistical-physics references <d-cite key="kschischang2001factor,pearl1988probabilistic,yedidia2005constructing,wainwright2008graphical,mezard2009information"></d-cite>. The diagrams retain editable TikZ sources. Quantitative figures are regenerated from human-readable data and small validation programs: tree marginals and $\log Z$ are compared with brute-force enumeration, and every displayed stability threshold is checked numerically against its defining fixed-point equation.
