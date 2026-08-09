---
layout: distill
title: Message Passing in Graph Neural Networks
description: What graph neural networks borrow from belief propagation, what they discard, and what the 1-WL ceiling says they can never see
img: assets/img/belief-propagation/wl-limit.png
permalink: /projects/belief-propagation/graph-neural-networks/
tags: graph-neural-networks message-passing weisfeiler-leman oversmoothing belief-propagation
importance: 99
category: work
show_on_projects: false
series: belief-propagation
series_part: 4
series_previous_url: /projects/belief-propagation/survey-propagation/
series_previous_label: "Part 3"
status: draft
giscus_comments: false
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: belief-propagation.bib
toc:
  - name: The same skeleton, different content
  - name: The MPNN abstraction
  - name: A taxonomy of aggregators
  - name: The expressiveness ceiling
  - name: Oversmoothing
  - name: Oversquashing
  - name: Where the probability survives
  - name: The five questions, answered honestly
---

{% include belief_propagation/series_nav.liquid %}

Parts 1 through 3 followed one construction: cut an edge, summarize everything on one side as a function of the boundary variable, pass the summary across the cut. Part 1 proved it exact on trees. Part 2 pushed it into optimization. Part 3 enlarged the message when a single marginal stopped being the right object.

Graph neural networks reuse the _shape_ of that computation and discard almost all of its content. A layer aggregates over neighbours and updates a node vector — structurally the same alternation Part 1 derived — but the aggregate and update are arbitrary learned functions, trained end-to-end on a downstream loss, with no joint distribution behind them.

This chapter is a survey rather than a derivation, because the interesting content here is a landscape of variants rather than a single theorem. But the reader arriving from Parts 1–3 has an advantage worth using: they already know what a message-passing scheme can and cannot buy, and can therefore ask sharper questions than "does it work well on benchmarks."

<div class="bp-callout warning">
<strong>The through-line.</strong> MPNNs generalize BP's computational template. They do not, in general, compute anything probabilistically meaningful. There is no normalization constraint, no partition function, no variational objective the layers provably descend, and no exactness theorem on trees — because there was never a $P(\mathbf s)$ to be exact about. Statements like "GNNs are just learned belief propagation" are true only of a specific narrow family, discussed at the end.
</div>

## The same skeleton, different content

It helps to put the two side by side before any architecture appears.

<table class="bp-decision-table">
<thead><tr><th></th><th>Belief propagation</th><th>Message-passing GNN</th></tr></thead>
<tbody>
<tr><td>Where the update comes from</td><td>Derived from a factorized $P(\mathbf s)$</td><td>Learned; chosen by architecture and loss</td></tr>
<tr><td>What a message is</td><td>A distribution (or its log/odds/zero-temperature limit)</td><td>An arbitrary real vector</td></tr>
<tr><td>Normalization</td><td>Required — a gauge, with beliefs summing to one</td><td>None required</td></tr>
<tr><td>Global objective</td><td>Bethe free energy; fixed points are stationary points</td><td>Task loss; no fixed-point interpretation</td></tr>
<tr><td>Iteration count</td><td>Run to convergence (a fixed point is the answer)</td><td>Fixed depth $T$, set as a hyperparameter</td></tr>
<tr><td>Exactness</td><td>Exact on trees (Part 1)</td><td>No analogue</td></tr>
<tr><td>Receptive field after $T$ steps</td><td>Depth-$T$ computation tree</td><td>The same depth-$T$ computation tree</td></tr>
</tbody>
</table>

The last row is the one that transfers completely, and it does real work later. A $T$-layer MPNN's output at node $i$ depends on exactly the depth-$T$ unrolling of nonbacktracking walks into $i$ — the same object Part 1 introduced to explain what loopy BP actually computes. Everything Part 1 said about that tree (a vertex reappearing at depth equal to its girth, boundary influence decaying or not) applies verbatim here, and reappears below under a different name.

## The MPNN abstraction

Gilmer and coauthors gave the unifying form: a message function $M^{(t)}$, a permutation-invariant aggregator $\bigoplus$, and an update $U^{(t)}$ <d-cite key="gilmer2017neural"></d-cite>. Per layer,

$$
\boxed{
m_i^{(t+1)}=\bigoplus_{j\in\mathcal N(i)} M^{(t)}\!\left(h_i^{(t)},h_j^{(t)},e_{ij}\right),
\qquad
h_i^{(t+1)}=U^{(t)}\!\left(h_i^{(t)},m_i^{(t+1)}\right).
}
$$

Set against Part 1's two boxed recursions, the correspondence is structural and partial. The aggregator $\bigoplus$ occupies the position of the factor-to-variable sum; the update $U$ occupies the position of the variable-to-factor product. But BP's product is forced — it follows from branch independence after pinning — whereas $U$ is whatever the architecture declares and training finds.

One difference is easy to miss and matters throughout. BP messages are _directed and exclude the recipient_: $\chi^{i\to a}$ omits $a$ precisely because $a$ is the cut. Standard MPNNs aggregate over all of $\mathcal N(i)$ with no exclusion, so node $i$'s own previous state flows back into it at every layer. That is not a bug — it is what residual-style updates want — but it means the MPNN computation tree contains backtracking walks that BP's construction deliberately removes.

<div class="bp-callout heuristic">
<strong>What transfers, precisely.</strong> The local-aggregate-then-update pattern; the depth-$T$ receptive field; the sensitivity of both to graph structure. <strong>What does not:</strong> normalization, a partition function, a fixed point, tree exactness, and any claim that $h_i^{(t)}$ estimates a marginal of anything.
</div>

## A taxonomy of aggregators

Nearly every well-known architecture is a choice of $\bigoplus$ and $U$. The useful way to read the table is as a chain of fixes: each row exists because of a specific limitation of an earlier one.

<table class="bp-decision-table">
<thead><tr><th>Model</th><th>Aggregation</th><th>Update</th><th>Introduced to fix</th></tr></thead>
<tbody>
<tr><td><strong>ChebNet</strong> <d-cite key="defferrard2016convolutional"></d-cite></td><td>Order-$K$ Chebyshev polynomial of the Laplacian — a spectral filter, not a per-neighbour message</td><td>Linear combination of Chebyshev bases, then nonlinearity</td><td>Made spectral graph convolution local and eigendecomposition-free</td></tr>
<tr><td><strong>GCN</strong> <d-cite key="kipf2017semi"></d-cite></td><td>Symmetric-normalized sum over $\mathcal N(i)\cup\{i\}$</td><td>Linear map, then nonlinearity</td><td>Collapsed ChebNet to one cheap first-order layer</td></tr>
<tr><td><strong>GraphSAGE</strong> <d-cite key="hamilton2017inductive"></d-cite></td><td>Mean / max-pool / LSTM over a fixed-size neighbour <em>sample</em></td><td>Concatenate self with aggregate, then linear</td><td>Inductive generalization to unseen nodes; scalability. GCN was transductive and full-graph</td></tr>
<tr><td><strong>GAT</strong> <d-cite key="velickovic2018graph"></d-cite></td><td>Sum weighted by learned softmax attention $\alpha_{ij}$</td><td>Weighted sum, multi-head concatenation</td><td>GCN weights neighbours by degree alone, treating all as equally informative</td></tr>
<tr><td><strong>GIN</strong> <d-cite key="xu2019powerful"></d-cite></td><td>Plain unweighted sum</td><td>$\mathrm{MLP}\big((1+\epsilon)h_i+\sum_{j\in\mathcal N(i)}h_j\big)$</td><td>Expressiveness: mean and max provably conflate multisets that sum can separate</td></tr>
<tr><td><strong>PNA</strong> <d-cite key="corso2020principal"></d-cite></td><td>Several aggregators in parallel (mean, max, min, std) with degree-dependent scalers</td><td>Concatenate all, then MLP</td><td>No single aggregator suffices for all multiset functions on continuous features</td></tr>
</tbody>
</table>

Three of these are worth writing out, because their differences are exactly the aggregator/update distinction above.

**GCN** normalizes by degree on both sides:

$$
h_i^{(t+1)}=\sigma\!\Big(W^{(t)}\!\!\sum_{j\in\mathcal N(i)\cup\{i\}}\tfrac{1}{\sqrt{d_id_j}}\,h_j^{(t)}\Big).
$$

**GAT** replaces the fixed $1/\sqrt{d_id_j}$ with a learned, softmax-normalized coefficient:

$$
\alpha_{ij}=\frac{\exp\!\big(\mathrm{LeakyReLU}(\mathbf a^\top[Wh_i\,\|\,Wh_j])\big)}
{\sum_{k\in\mathcal N(i)}\exp\!\big(\mathrm{LeakyReLU}(\mathbf a^\top[Wh_i\,\|\,Wh_k])\big)},
\qquad
h_i^{(t+1)}=\sigma\!\Big(\sum_{j\in\mathcal N(i)}\alpha_{ij}Wh_j^{(t)}\Big).
$$

That softmax is the closest thing in this chapter to BP's normalization — and it is worth being clear that the resemblance is superficial. BP normalizes a message so that it is a distribution over the _states of one variable_. GAT normalizes attention weights across the _neighbours of one node_. Different index, different meaning; the constraint carries no probabilistic semantics about $h$.

**GIN** does the opposite of normalizing, deliberately:

$$
h_i^{(t+1)}=\mathrm{MLP}^{(t)}\!\Big((1+\epsilon^{(t)})\,h_i^{(t)}+\sum_{j\in\mathcal N(i)}h_j^{(t)}\Big).
$$

The unweighted sum is the point. A mean forgets multiplicity — $\{a,a,b\}$ and $\{a,b\}$ have the same mean — and a max forgets everything but the extremes. A sum composed with an injective MLP can, in principle, distinguish any finite multiset, and that is precisely what the next section needs.

### What the aggregator choice costs

The taxonomy is easier to hold in mind as a set of trade-offs than as a list of papers.

<table class="bp-decision-table">
<thead><tr><th>Choice</th><th>Buys</th><th>Costs</th></tr></thead>
<tbody>
<tr><td>Degree normalization (GCN)</td><td>Stable scales across wildly varying degrees; a well-conditioned operator</td><td>Multiplicity information; strictly below the 1-WL ceiling</td></tr>
<tr><td>Neighbour sampling (GraphSAGE)</td><td>Bounded cost per node regardless of degree; inductive use on unseen graphs</td><td>Stochastic aggregation; the LSTM variant is not permutation-invariant without extra care</td></tr>
<tr><td>Learned attention (GAT)</td><td>Non-uniform neighbour weighting; some interpretability</td><td>Extra parameters; no expressiveness gain in the worst case</td></tr>
<tr><td>Plain sum (GIN)</td><td>Multiset injectivity; matches the 1-WL ceiling</td><td>Scale grows with degree; can be numerically awkward on heavy-tailed graphs</td></tr>
<tr><td>Multiple aggregators (PNA)</td><td>Complementary statistics; better on continuous features</td><td>Wider layers; more compute per edge</td></tr>
</tbody>
</table>

Two observations a reader from Parts 1–3 is well placed to make. First, none of these choices is derived — each is a design decision validated empirically, whereas BP's product-and-sum was forced by the factorization. Second, the trade-offs are recognizably the same ones Part 2 discussed for high-arity factors: you can have cheap updates or expressive updates, and buying the second costs compute in a predictable way.

## The expressiveness ceiling

The sharpest result about message passing is a negative one, and it has a clean combinatorial statement.

**1-dimensional Weisfeiler–Leman colour refinement** assigns every node an initial colour, then repeatedly recolours:

$$
c^{(t+1)}(v)=\mathrm{hash}\!\Big(c^{(t)}(v),\ \{\!\{c^{(t)}(u):u\in\mathcal N(v)\}\!\}\Big),
$$

where $\{\!\{\cdot\}\!\}$ is a multiset. Run to a fixed point; declare two graphs _distinguishable_ if their final colour multisets differ. This is a classical graph-isomorphism heuristic — sound but incomplete.

The result of Xu et al. and Morris et al. is that this heuristic is exactly the ceiling for message passing: after $T$ layers, any MPNN's node representations are at most as discriminative as $T$ rounds of 1-WL, and an MPNN with injective aggregation and update — GIN — attains that bound <d-cite key="xu2019powerful,morris2019weisfeiler"></d-cite>. Architectures with non-injective aggregators (GCN's normalized mean, GraphSAGE's mean or max) sit strictly below it.

{% include figure.liquid path="assets/img/belief-propagation/wl-limit.svg" alt="Two disjoint triangles beside a six-cycle; both have six nodes of degree two and identical Weisfeiler-Leman colour multisets" caption="The canonical failure. Both graphs have six nodes, all of degree two, so every round of 1-WL assigns every node the same colour in both — the colour multisets are identical at every round. Yet $G_1$ contains two triangles and $G_2$ contains none, and they are not isomorphic. No message-passing GNN with uninformative initial features can separate them, GIN included." %}

<div class="bp-callout derivation">
<strong>Checkable by hand.</strong> Take $G_1=$ two disjoint triangles and $G_2=$ one 6-cycle. Every node in both graphs has degree $2$, so with uniform initial colours the first refinement round gives every node the same $(\text{colour},\{\!\{\text{colour},\text{colour}\}\!\})$ signature — in both graphs. That is a fixed point, so refinement stops with a single colour class of size six on each side. The multisets are equal; 1-WL fails. Counting triangles separates them instantly ($2$ versus $0$), which is the point: triangle counting is not something local message passing can do.
</div>

The route past the ceiling is to change the object being refined rather than the training: $k$-dimensional WL refines colours on $k$-tuples of nodes, and the corresponding $k$-GNNs are strictly more expressive <d-cite key="morris2019weisfeiler"></d-cite>. The cost is the familiar one from Part 1's remark on high-arity factors: tracking $k$-tuples is combinatorial in $k$, so expressiveness is bought with computation, not for free.

<div class="bp-callout warning">
<strong>Scope of the ceiling.</strong> "1-WL bounds GNNs" is correct for <em>plain message-passing</em> GNNs under the standard feature setup. It is not a statement about all graph architectures. Higher-order GNNs, subgraph-based methods, and message passing augmented with structural or positional features (distance encodings, Laplacian eigenvectors, random identifiers) can and do exceed it. The frequently repeated "GNNs cannot exceed 1-WL" drops both qualifiers.
</div>

### Why sum beats mean: a two-line argument

GIN's choice of aggregator is the one place in this chapter where an architectural decision follows from a proof rather than an experiment, so it is worth seeing.

Suppose a node's neighbourhood is described by a _multiset_ of neighbour features — multiplicity matters, order does not. Consider two neighbourhoods

$$
\mathcal A=\{\!\{a,a,b\}\!\},\qquad \mathcal B=\{\!\{a,b\}\!\}\quad\text{extended to equal size by }\{\!\{a,b,b\}\!\}.
$$

A **mean** aggregator maps $\{\!\{a,a,b\}\!\}\mapsto\frac{2a+b}{3}$ and $\{\!\{a,b\}\!\}\mapsto\frac{a+b}{2}$; for the specific case $\{\!\{a,a\}\!\}$ versus $\{\!\{a\}\!\}$ both give exactly $a$, so multiplicity is destroyed outright. A **max** aggregator keeps only the extreme element, so $\{\!\{a,a,b\}\!\}$ and $\{\!\{a,b,b\}\!\}$ are identical whenever $b$ dominates. A **sum** keeps $2a+b$ versus $a+2b$ — distinct whenever $a\ne b$.

That is the entire argument, and its consequence is the GIN update: sum to preserve the multiset, then apply an MLP that can in principle realize an injective map on the resulting vectors <d-cite key="xu2019powerful"></d-cite>. Every other row of the taxonomy table trades some of that injectivity for something else — degree normalization for stability, attention for selectivity, sampling for scalability.

<div class="bp-callout warning">
<strong>"In principle" is doing work.</strong> The injectivity argument is about what the architecture <em>can</em> represent, not what training finds. A GIN with finite width and a particular initialization is not guaranteed to realize an injective map, and the theorem says nothing about generalization. Maximal expressiveness within the 1-WL class is a statement about the hypothesis space.
</div>

### Beyond the ceiling, and what it costs

If 1-WL is the ceiling, the obvious question is how to get above it. Three routes are standard, and they price differently.

**Refine on tuples.** $k$-WL colours $k$-tuples of nodes rather than nodes, and the corresponding $k$-GNNs are strictly more expressive as $k$ grows <d-cite key="morris2019weisfeiler"></d-cite>. The cost is immediate: there are $n^k$ tuples, so memory and time grow polynomially with a rapidly increasing exponent. This is the same bargain Part 2 described for high-arity factors — a wider local scope buys discrimination and is paid for in combinatorics.

**Break the symmetry with features.** Give nodes something to distinguish them: distance encodings, Laplacian eigenvectors, random identifiers. The triangle/6-cycle pair becomes separable the moment nodes carry a triangle count, because the information 1-WL cannot derive is supplied as input. This is cheap and effective, and it relocates rather than removes the problem — one must now argue the chosen features are computable and meaningful for the task.

**Look at subgraphs.** Represent a graph by a collection of its subgraphs and aggregate over them, which recovers information about local structure that node-level refinement discards.

<div class="bp-callout heuristic">
<strong>The honest framing.</strong> "GNNs are limited by 1-WL" describes a specific, standard setup: message passing over node features with uninformative initialization. Every route above escapes it, and none is free. Quoting the ceiling without the qualifier makes a precise theorem sound like a law of nature.
</div>

## Oversmoothing

Depth in an MPNN does not behave like depth in a convolutional network. Stacking layers enlarges the receptive field, but repeated neighbourhood averaging also drives node representations together.

Li, Han and Wu identified the mechanism: GCN's propagation is a form of Laplacian smoothing, and iterating a smoothing operator drives features toward its dominant eigenvector, so representations of distinct nodes converge <d-cite key="li2018deeper"></d-cite>. Oono and Suzuki sharpened this into an asymptotic statement — under stated conditions on the weights and the graph spectrum, the component of the representation that distinguishes nodes decays exponentially in depth <d-cite key="oono2020graph"></d-cite>.

For a reader coming from Parts 1–3, the important thing is that **this is not BP's loopy-graph error**. BP's error came from a mismatch between the computation tree and the real graph — correlated evidence counted as independent. Oversmoothing is a statement about repeatedly applying a fixed contraction, closer in spirit to a Markov chain forgetting its initial condition than to anything in Part 1. The two can coexist in the same model, and confusing them leads to the wrong fix.

<div class="bp-callout warning">
<strong>Not "deep GNNs don't work."</strong> The Oono–Suzuki result is asymptotic and conditional. It does not say every deep GNN degenerates in practice, and residual connections, normalization, and non-averaging aggregators change the picture substantially. State it the way Part 2 states infinite-tree thresholds: as an asymptotic result about a model class under hypotheses, not a universal empirical law.
</div>

## Oversquashing

The second depth pathology is about capacity rather than collapse, and it is where Part 1's computation tree earns its keep.

A node's depth-$T$ receptive field grows like the branching factor to the $T$-th power, but its representation stays a fixed-width vector. Information from an exponentially large neighbourhood is compressed into constant space, and long-range dependencies get crushed. Alon and Yahav named this **oversquashing** and showed it bites precisely on tasks needing long-range interaction <d-cite key="alon2021bottleneck"></d-cite>.

Topping and coauthors made the graph-theoretic cause precise: the sensitivity $\partial h_i^{(T)}/\partial h_j^{(0)}$ of a node's output to a distant input is controlled by a discrete Ricci-type curvature of the edges along the connecting paths, with negatively curved bottleneck edges throttling the signal <d-cite key="topping2022understanding"></d-cite>. That turns a diagnosis into a prescription: rewire the graph to relieve the identified bottlenecks.

<div class="bp-callout heuristic">
<strong>Two distinct failures, routinely conflated.</strong> Oversmoothing is representations of <em>different nodes</em> converging to one another as depth grows. Oversquashing is <em>long-range information</em> failing to survive compression through a bottleneck, and it appears at moderate depth on graphs with the wrong topology. They interact, but their fixes differ: normalization and residual connections for the first, rewiring and curvature-aware architectures for the second.
</div>

Both, however, are statements about the same object Part 1 introduced. Oversmoothing is what happens when the computation tree's boundary influence decays too fast; oversquashing is what happens when too much of that tree must fit through too narrow a channel. The unrolling that explained loopy BP's error explains both of message passing's depth pathologies — the mechanisms differ, the geometry is shared.

### The Jacobian view, and why it connects to Part 1

Topping and coauthors' formulation is worth writing down because it makes the bottleneck quantitative rather than metaphorical. The influence of a distant input on a node's output is measured by

$$
\left\lVert\frac{\partial h_i^{(T)}}{\partial h_j^{(0)}}\right\rVert,
$$

and this quantity is bounded above by a product of terms along the paths from $j$ to $i$ — terms that shrink where the graph is negatively curved, i.e. where many shortest paths funnel through few edges <d-cite key="topping2022understanding"></d-cite>.

Compare Part 1. There, the influence of the depth-$t$ boundary of the computation tree on its root controlled whether loopy BP's answer was trustworthy: decaying influence meant the surrogate forgot its boundary condition and the fixed point was meaningful. Here, influence decaying _too fast_ along a path is the pathology, because the task needs that distant information to arrive.

Same object, opposite desiderata. Inference wants boundary influence to vanish, so that the answer does not depend on an arbitrary initialization. Learning wants it to survive, so that a label depending on a far-away subgraph is computable at all. A graph with strong bottlenecks is good news for the first and bad news for the second — which is a genuinely useful thing to notice when moving between the two literatures.

<div class="bp-callout heuristic">
<strong>What rewiring does and does not fix.</strong> Adding edges to relieve a bottleneck improves the Jacobian bound, and empirically improves long-range tasks. It also changes the graph the model sees, which for some problems is the object of study rather than a nuisance. Rewiring a molecule's bond graph to help optimization is a modelling decision that needs its own justification, not a free preprocessing step.
</div>

## Where the probability survives

Not every learned message-passing method throws away the probabilistic structure. Three families keep progressively more of it, and they are the honest answer to "are GNNs learned BP?"

**Learned BP for decoding.** Nachmani, Be'ery and Burshtein keep the sum-product update on a fixed Tanner graph and attach trainable weights to its edges, unrolling a fixed number of iterations into a network <d-cite key="nachmani2016learning"></d-cite>. The messages remain log-likelihood ratios; the schedule remains BP's. Only the edge weights are learned, compensating for the short cycles that make plain BP suboptimal on finite codes.

**Neural-enhanced BP.** Satorras and Welling run genuine BP on the factor graph and use a GNN as a learned correction alongside it, combining the two message streams <d-cite key="satorras2021neural"></d-cite>. The factor-graph structure and BP's update survive intact; the network supplies what BP's approximation omits.

**Belief propagation neural networks.** Kuck and coauthors construct a strict generalization whose parameters can be set to recover ordinary BP exactly, so the learned model contains BP as a special case rather than merely resembling it <d-cite key="kuck2020belief"></d-cite>.

<div class="bp-callout exact">
<strong>The distinction that matters.</strong> These three are BP-with-learning: the factor graph, the message semantics, and in the last case BP itself as a reachable special case, all survive. GCN, GAT and GIN retain none of that — only the topology and the aggregation pattern. "Neural networks discovered belief propagation" inverts the history: these architectures are <em>initialized</em> on BP's structure by construction. The interesting quantity is how much correction the learned part contributes on top, which should be reported as a magnitude.
</div>

### A spectrum, not a dichotomy

It is tempting to sort methods into "principled BP" and "black-box GNN". The three families above show the space is continuous, and it is more useful to ask _how much structure is retained_.

<table class="bp-decision-table">
<thead><tr><th>Method</th><th>Factor graph</th><th>Message semantics</th><th>Recovers BP exactly?</th><th>What is learned</th></tr></thead>
<tbody>
<tr><td>Plain BP</td><td>Given by the model</td><td>Distributions</td><td>—</td><td>Nothing</td></tr>
<tr><td>Learned-weight BP <d-cite key="nachmani2016learning"></d-cite></td><td>Fixed (Tanner graph)</td><td>Log-likelihood ratios</td><td>Yes, at unit weights</td><td>Per-edge scalars</td></tr>
<tr><td>Neural-enhanced BP <d-cite key="satorras2021neural"></d-cite></td><td>Retained</td><td>BP messages plus a learned channel</td><td>Yes, if the correction vanishes</td><td>A correction network</td></tr>
<tr><td>BP neural networks <d-cite key="kuck2020belief"></d-cite></td><td>Retained</td><td>Generalized BP messages</td><td>Yes, by construction</td><td>Update parameters</td></tr>
<tr><td>GIN / GCN / GAT</td><td>Only the graph topology</td><td>Uninterpreted vectors</td><td>No</td><td>Everything</td></tr>
</tbody>
</table>

The column that matters is the fourth. A method that contains BP as a special case can be _no worse_ than BP after training, at least in principle, because the optimizer can always fall back. A method that cannot represent BP has no such floor — it may do far better on a task BP was never suited to, and far worse on one BP solves exactly.

That framing also suggests the honest experiment for anyone claiming a learned method beats BP: check whether the architecture can represent BP at all. If it can, report how far from BP the learned solution ended up. If it cannot, the comparison is between different objects and the win may be a statement about the task rather than the method.

## The five questions, answered honestly

The series opened with five questions to ask of any message-passing method. For a generic trained MPNN the answers are humbling, and that is the useful part.

1. **What global quantity is being computed?** Nothing fixed. Whatever the task loss defines — a node label, a graph property, a molecular energy. There is no $Z$, no marginal, no free energy.
2. **What does one message summarize?** Whatever training made useful. Unlike a BP message, it has no independent definition, so it cannot be checked against anything except downstream performance.
3. **Where does factorization enter?** Only as the fixed 1-hop locality of each layer. There is no claim that neighbours are conditionally independent — the architecture never needed one, because it never had a joint distribution to factor.
4. **What does convergence mean?** Nothing. An MPNN runs a fixed depth $T$; it is not iterated to a fixed point, and stacking more layers makes things worse for the two reasons above rather than better.
5. **What is independently checkable?** Expressive-power theory — the 1-WL ceiling and its refinements — plus explicit constructions like the triangle/6-cycle pair. That is a genuine, architecture-level check, and it is the closest analogue in this chapter to Part 1's "does it return $108$ on four variables."

That last asymmetry is the honest summary of the whole chapter. Parts 1–3 could always fall back on an exact small instance: enumerate it, compare, and know. Part 4 cannot, because there is no ground-truth quantity a general MPNN is trying to compute. What replaces it is a different kind of rigour — combinatorial statements about what a model class can and cannot distinguish, proved once and applying to every trained instance.

Both are worth having. A field that only had the first would never have built architectures this flexible; a field that only had the second would have no way to tell a genuine advance from a better-tuned baseline.

### What a reader should take from four chapters of message passing

The series has now shown the same computational pattern in four settings, and the differences between them are more instructive than the similarity.

<table class="bp-decision-table">
<thead><tr><th>Chapter</th><th>Message carries</th><th>Justified by</th><th>Checkable against</th></tr></thead>
<tbody>
<tr><td>1 — trees</td><td>A pinned subtree partition function</td><td>A proof</td><td>Exact enumeration</td></tr>
<tr><td>2 — optimization</td><td>Costs, odds, or a zero-temperature score</td><td>A proof on trees; an approximation on loops</td><td>Enumeration, plus thresholds derived in closed form</td></tr>
<tr><td>3 — survey propagation</td><td>A distribution over messages, one per cluster</td><td>A theorem for the algorithm; a prediction for the interpretation</td><td>Small shattered instances; one proved threshold</td></tr>
<tr><td>4 — neural message passing</td><td>An uninterpreted learned vector</td><td>Empirical performance</td><td>Expressive-power theory and explicit counterexamples</td></tr>
</tbody>
</table>

Reading down the third column is the actual arc: the justification weakens at every step, and the thing being computed becomes less well-defined. That is not a decline. Each step buys something the previous one could not do — optimization, shattered landscapes, arbitrary learned tasks — and pays for it in guarantees.

What should stay constant is the habit of asking which column you are in. A result reported as though it belonged in row one, when it belongs in row four, is the single most common way this literature is misread.

That closes the arc this series set out to trace. Belief propagation began as exact dynamic programming on a tree, became a controlled approximation on loopy graphs, needed a larger message when the solution space shattered, and finally lent its computational skeleton to a family of models that kept the shape and abandoned the semantics. The equations look similar throughout. What changes each time — and what is worth asking first — is what the messages are _for_.
