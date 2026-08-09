---
layout: distill
title: Belief Propagation from First Principles
description: Cut a tree, pin its boundary, and derive sum–product and the Bethe variational principle without guessing the update rules
img: assets/img/belief-propagation/worked-message-sweep.png
permalink: /projects/belief-propagation/from-first-principles/
tags: graphical-models belief-propagation sum-product bethe-free-energy
importance: 99
category: work
show_on_projects: false
series: belief-propagation
series_part: 1
series_next_url: /projects/belief-propagation/combinatorial-optimization/
series_next_label: "Part 2"
status: draft
giscus_comments: false
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: belief-propagation.bib
toc:
  - name: The marginalization problem
  - name: A tree small enough to expand
  - name: Cut an edge and pin the boundary
  - name: Derive the two recursions
  - name: Normalize into messages
  - name: Recover marginals and the partition function
  - name: The variational meaning of BP
  - name: What loopy BP actually computes
  - name: Living with loops
  - name: What has actually been proved
---

{% include belief_propagation/series_nav.liquid %}

Belief propagation is usually introduced as two update equations. That is efficient for implementation and poor for understanding. Why is there a product in one direction and a sum in the other? What information does a single message contain? Why is the method exact on a tree, and precisely which statement stops being true when one loop is added?

This chapter derives the algorithm rather than announcing it. The central object is not initially a probability message but a **pinned subtree partition function**: cut an edge, hold the boundary variable fixed, and sum everything on one side. On a tree the cut disconnects the graph, and the sum–product equations are exactly the recursion those partial sums obey. This dynamic-programming reading is the shared core behind Pearl's tree algorithms and the factor-graph sum–product formalism <d-cite key="pearl1988probabilistic,kschischang2001factor"></d-cite>.

Everything is developed on a running four-variable example with explicit integers, so each claimed identity can be checked by hand. At the end we ask what survives on a loopy graph, and answer with a precise object — the computation tree — rather than an intuition.

<div class="bp-callout exact">
<strong>Exact statement.</strong> On a finite tree factor graph, two directed messages per edge determine every variable and factor marginal and the exact partition function. The proof uses only distributivity and the fact that removing an edge disconnects a tree.
</div>

## The marginalization problem

Let the variables be $s_i\in\Lambda_i$ with every alphabet finite. A factor graph has variable nodes $i\in V$, factor nodes $a\in F$, and an edge $(i,a)$ whenever factor $a$ depends on $s_i$. Write $\partial i$ for the factors adjacent to variable $i$, $\partial a$ for the variables adjacent to factor $a$, and $d_i=\lvert\partial i\rvert$. Allow a unary weight $g_i(s_i)\ge0$ at each variable and a nonnegative interaction $f_a(\mathbf s_{\partial a})\ge0$ at each factor:

$$
P(\mathbf s)
=\frac{1}{Z}
\prod_{i\in V}g_i(s_i)
\prod_{a\in F}f_a(\mathbf s_{\partial a}),
\qquad
Z=\sum_{\mathbf s}
\prod_i g_i(s_i)\prod_a f_a(\mathbf s_{\partial a}).
$$

{% include figure.liquid path="assets/img/belief-propagation/factor-graph-anatomy.svg" alt="A bipartite factor graph with circular variable nodes, square factor nodes, neighborhood labels, and the factorized probability law" caption="A factor graph records which variables enter each local function. An edge means 'is an argument of'; it is not by itself a claim of marginal dependence." %}

Two quantities matter throughout. The **marginal**

$$
P_i(s_i)=\sum_{\mathbf s_{V\setminus i}}P(\mathbf s)
$$

answers local questions, and the **partition function** $Z$ answers global ones — counting, free energy, model comparison, and the normalization every marginal secretly depends on.

A direct evaluation visits $\prod_i\lvert\Lambda_i\rvert$ assignments; with a common alphabet of size $q$ that is $q^{\lvert V\rvert}$. Factorization alone does not remove this cost. A dense or highly loopy factor graph can encode a genuinely hard counting problem, and no rearrangement of the same sum will rescue it.

### Why elimination order is the whole game

The useful structure is not factorization but factorization _with small separators_. Summing out one variable at a time is variable elimination: each eliminated variable is replaced by a new factor over its remaining neighbors. The cost is governed by the largest intermediate factor produced, and minimizing that over orderings defines the treewidth of the graph <d-cite key="koller2009probabilistic"></d-cite>.

A chain admits an order in which every intermediate factor has one free variable. A tree does too. A grid does not: eliminating along a row leaves a factor whose scope grows with the row length. This is the honest statement of when message passing helps — the graph must admit an elimination order with bounded intermediate scope, and trees are the case where that order is obvious and the scope is one.

<table class="bp-notation">
<thead><tr><th>Symbol</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td>$\partial i,\ \partial a$</td><td>neighbors of variable $i$ and of factor $a$; $d_i=\lvert\partial i\rvert$</td></tr>
<tr><td>$R^{i\to a}_{s_i}$</td><td>partition function on the variable side of cut $(i,a)$, with $s_i$ pinned</td></tr>
<tr><td>$V^{a\to i}_{s_i}$</td><td>partition function on the factor side of the same cut, with $s_i$ pinned</td></tr>
<tr><td>$\chi^{i\to a},\ \psi^{a\to i}$</td><td>normalized versions of $R^{i\to a}$ and $V^{a\to i}$</td></tr>
<tr><td>$b_i,\ b_a$</td><td>variable and factor beliefs reconstructed from messages</td></tr>
<tr><td>$Z^i,\ Z^a,\ Z^{ia}$</td><td>local normalizers at a variable, a factor, and an incidence</td></tr>
</tbody>
</table>

## A tree small enough to expand

Take four variables and three pair factors:

$$
W(s_1,s_2,s_3,s_4)
=g_1(s_1)g_2(s_2)g_3(s_3)g_4(s_4)\,
 f_a(s_1,s_2)\,f_b(s_2,s_3)\,f_c(s_2,s_4).
$$

The partition function is the fourfold sum $Z=\sum_{s_1,s_2,s_3,s_4}W$. Rather than enumerate quadruples, push each sum as far right as it will go:

$$
\begin{aligned}
Z
&=\sum_{s_1}g_1(s_1)\sum_{s_2}f_a(s_1,s_2)g_2(s_2)\\
&\quad\times
\left[\sum_{s_3}f_b(s_2,s_3)g_3(s_3)\right]
\left[\sum_{s_4}f_c(s_2,s_4)g_4(s_4)\right].
\end{aligned}
$$

The bracketed quantities are functions of $s_2$ alone. Each summarizes an entire branch, and once tabulated the branch variables never reappear. Nothing here is specific to probability: it is distributivity plus an elimination order in which intermediate functions never acquire a wide scope.

{% include figure.liquid path="assets/img/belief-propagation/worked-message-sweep.svg" alt="Four copies of a small tree showing leaf, factor, middle-variable, and root stages of an inward message sweep" caption="An inward sweep evaluates the global sum from the leaves. Each arrow replaces a complete branch by a function of the single variable at its boundary." %}

### The same tree with numbers in it

Abstract recursions are easy to nod at and hard to check. Fix binary variables $s_i\in\{0,1\}$ and these integer weights:

$$
g_1=\begin{pmatrix}1\\1\end{pmatrix},\;
g_2=\begin{pmatrix}1\\2\end{pmatrix},\;
g_3=\begin{pmatrix}1\\1\end{pmatrix},\;
g_4=\begin{pmatrix}3\\1\end{pmatrix},
$$

$$
f_a=\begin{pmatrix}2&1\\1&2\end{pmatrix},\quad
f_b=\begin{pmatrix}3&1\\1&1\end{pmatrix},\quad
f_c=\begin{pmatrix}1&1\\1&2\end{pmatrix},
$$

where $f_a$ has rows indexed by $s_1$ and columns by $s_2$, and $f_b,f_c$ have rows indexed by $s_2$. Brute-force enumeration of all sixteen assignments gives

$$
Z=108 .
$$

We will recompute this number three different ways, and every intermediate quantity below is an exact integer or fraction. Keep the value $108$ in mind; it is the anchor for the rest of the chapter.

For alphabet size $q$ a pair-factor update costs $O(q^2)$ instead of the $O(q^4)$ full enumeration. A factor of arity $r$ costs $O(q^r)$ under the naive update. BP is linear in the number of edges only when alphabet size and factor arity are treated as bounded constants — a caveat worth stating explicitly, because high-arity factors are common and their updates need dedicated structure to stay cheap.

## Cut an edge and pin the boundary

Take a directed edge $i\to a$. Delete $(i,a)$ and look at the connected component containing $i$. Define

$$
R^{i\to a}_{s_i}=\!\!\sum_{\text{variables in that component except }i}\ \prod_{\text{weights inside it}},
$$

holding $s_i$ fixed. Symmetrically, $V^{a\to i}_{s_i}$ sums the component containing $a$ with $s_i$ fixed where it enters $f_a$.

These definitions are semantic, not yet algorithmic. A message answers one question:

> If the rest of the graph asks this component to use boundary value $s_i$, what total weight can the component supply?

{% include figure.liquid path="assets/img/belief-propagation/tree-cut-recursion.svg" alt="A tree cut between a variable and an adjacent factor, with the variable-side subtree highlighted and its two independent branches visible" caption="Deleting one tree edge disconnects the graph. After the boundary value at the cut is pinned, the highlighted branches contain disjoint variables, so their partial partition functions multiply exactly. The figure labels the cut variable $j$ and the excluded factor $a$; the text above uses $i$ for the same role." %}

Pinning is what makes this work. Without fixing $s_i$, the branches meeting at $i$ stay coupled, because they must agree on a shared value. Conditional on $s_i$, no variable is shared between distinct branches of a tree — the branches are separated by the single node $i$. In graphical-model language, one variable is a separator; in physics language, fixing $s_i$ creates a cavity in which the branches are independent. This exact conditional independence is the one ingredient that becomes an approximation later.

Note what the definition does _not_ require: no assumption about correlation strength, no limit, no ansatz. On a tree, $R$ and $V$ are well-defined finite sums, and the only question is how to compute them.

## Derive the two recursions

### Variable to factor: independent branches multiply

In the component behind $i\to a$, variable $i$ contributes $g_i(s_i)$. Each adjacent factor $b\in\partial i\setminus a$ roots a branch, those branches are pairwise disjoint once $s_i$ is pinned, and each contributes $V^{b\to i}_{s_i}$. A sum over a product of terms with disjoint variable sets factorizes into a product of sums, so

$$
\boxed{
R^{i\to a}_{s_i}
=g_i(s_i)
\prod_{b\in\partial i\setminus a}V^{b\to i}_{s_i}.
}
$$

No sum appears because $s_i$ is still pinned. The factor $a$ is excluded because that is precisely the edge removed to define the component.

### Factor to variable: eliminate the factor's internal boundary

Now consider $a\to i$ and keep $s_i$ fixed. The other variables $s_j$ for $j\in\partial a\setminus i$ are internal to this component and must be summed out. For each joint assignment of them, the local factor multiplies the independent variable-side branch weights:

$$
\boxed{
V^{a\to i}_{s_i}
=
\sum_{\mathbf s_{\partial a\setminus i}}
 f_a(\mathbf s_{\partial a})
 \prod_{j\in\partial a\setminus i}R^{j\to a}_{s_j}.
}
$$

This is the origin of the name **sum–product**. Products assemble independent components at a fixed boundary; sums eliminate internal variables. The asymmetry between the two updates is not a convention — it reflects that variable nodes join branches that must agree, while factor nodes join variables that may differ.

<div class="bp-callout derivation">
<strong>The running example, from the leaves inward.</strong>
Leaves first:
$$R^{3\to b}=(1,1),\qquad R^{4\to c}=(3,1).$$
Factor messages into variable $2$:
$$V^{b\to2}_{s_2}=\sum_{s_3}f_b(s_2,s_3)g_3(s_3)=(4,2),$$
$$V^{c\to2}_{s_2}=\sum_{s_4}f_c(s_2,s_4)g_4(s_4)=(4,5).$$
The middle variable multiplies its two branches with its own weight:
$$R^{2\to a}_{s_2}=g_2(s_2)V^{b\to2}_{s_2}V^{c\to2}_{s_2}=(1\cdot4\cdot4,\;2\cdot2\cdot5)=(16,20).$$
The last factor sums over $s_2$:
$$V^{a\to1}_{s_1}=\sum_{s_2}f_a(s_1,s_2)R^{2\to a}_{s_2}=(2\cdot16+1\cdot20,\;1\cdot16+2\cdot20)=(52,56).$$
Finally
$$Z=\sum_{s_1}g_1(s_1)V^{a\to1}_{s_1}=52+56=108 .$$
</div>

That is the same $108$ produced by enumerating sixteen assignments, obtained with four small updates. The general proof is induction on subtree depth: leaves satisfy the definitions directly; if every child message equals its pinned subtree sum, then branch disjointness plus distributivity gives the parent message. Working inward reaches any chosen root.

## Normalize into messages

Unnormalized $R$ and $V$ values grow or shrink exponentially with subtree size — on a chain of a few hundred variables they overflow or underflow double precision. But scaling an entire directed message by a positive constant changes neither the normalized beliefs nor any subsequent message ratio. So define

$$
\chi^{i\to a}_{s_i}=\frac{R^{i\to a}_{s_i}}{\sum_tR^{i\to a}_t},
\qquad
\psi^{a\to i}_{s_i}=\frac{V^{a\to i}_{s_i}}{\sum_tV^{a\to i}_t},
$$

which turns the recursions into the familiar equations

$$
\boxed{
\chi^{i\to a}_{s_i}
\propto g_i(s_i)
\prod_{b\in\partial i\setminus a}\psi^{b\to i}_{s_i}
}
$$

$$
\boxed{
\psi^{a\to i}_{s_i}
\propto
\sum_{\mathbf s_{\partial a\setminus i}}
 f_a(\mathbf s_{\partial a})
 \prod_{j\in\partial a\setminus i}\chi^{j\to a}_{s_j}.
}
$$

Each proportionality constant is chosen independently per directed edge so the entries sum to one. This is a **gauge choice**, not a modelling assumption. Other gauges are perfectly legal — fixing $\max_s m(s)=1$, or fixing one log-message entry to zero. The consequence is practical: two implementations can hold numerically different message vectors and represent the same fixed point. Convergence tests and equality checks must therefore compare gauge-invariant beliefs, or compare messages only after imposing a common normalization.

In the log domain one stores $\mu=\log m$ and replaces the variable update by a sum and the factor update by a log-sum-exp with the maximum pulled out:

$$
\log\sum_k e^{\mu_k}=\mu_{\max}+\log\sum_k e^{\mu_k-\mu_{\max}} .
$$

This is the standard defence against underflow and costs nothing in accuracy. Part 2 shows that the same substitution, taken to its extreme, produces min-sum.

## Recover marginals and the partition function

Once every incoming direction is available, the variable belief is

$$
b_i(s_i)
=\frac{1}{Z^i}
 g_i(s_i)\prod_{a\in\partial i}\psi^{a\to i}_{s_i},
\qquad
Z^i=\sum_{s_i}g_i(s_i)\prod_{a\in\partial i}\psi^{a\to i}_{s_i},
$$

and the factor belief is

$$
b_a(\mathbf s_{\partial a})
=\frac{1}{Z^a}
f_a(\mathbf s_{\partial a})
\prod_{i\in\partial a}\chi^{i\to a}_{s_i},
\qquad
Z^a=\sum_{\mathbf s_{\partial a}}f_a(\mathbf s_{\partial a})\prod_{i\in\partial a}\chi^{i\to a}_{s_i}.
$$

At any fixed point these are locally consistent:

$$
\sum_{\mathbf s_{\partial a\setminus i}}b_a(\mathbf s_{\partial a})=b_i(s_i).
$$

### One sweep is not enough

A single inward pass toward a root $r$ delivers $b_r$ and $Z$ — and nothing else. Every other variable has an incoming message on only one side. To obtain all marginals, run the complementary outward pass, so that each directed edge is evaluated once in each direction. On a chain this is exactly the forward–backward algorithm: the inward pass is the forward recursion, the outward pass the backward one, and their pointwise product is the smoothed posterior. Belief propagation on a tree is forward–backward with the chain replaced by a branching structure.

<div class="bp-callout derivation">
<strong>Outward pass on the running example.</strong>
The root sends $R^{1\to a}=(1,1)$, so
$$V^{a\to2}_{s_2}=\sum_{s_1}f_a(s_1,s_2)g_1(s_1)=(3,3).$$
Now variable $2$ has all three incoming messages:
$$b_2\propto g_2\cdot V^{a\to2}\cdot V^{b\to2}\cdot V^{c\to2}=(1\cdot3\cdot4\cdot4,\;2\cdot3\cdot2\cdot5)=(48,60),$$
so $b_2=\left(\tfrac49,\tfrac59\right)$. The root belief is $b_1=(52,56)/108=\left(\tfrac{13}{27},\tfrac{14}{27}\right)$.
Brute-force enumeration returns exactly these fractions.
</div>

### The partition function from local normalizers

Define the incidence overlap

$$
Z^{ia}=\sum_{s_i}\chi^{i\to a}_{s_i}\,\psi^{a\to i}_{s_i}.
$$

Then on a tree

$$
\boxed{
Z=
\frac{\displaystyle\prod_i Z^i\prod_a Z^a}
{\displaystyle\prod_{(i,a)}Z^{ia}}
\qquad\Longleftrightarrow\qquad
\log Z=\sum_i\log Z^i+\sum_a\log Z^a-\sum_{(i,a)}\log Z^{ia}.
}
$$

Evaluated on the running example this expression returns $108$ exactly — a third independent route to the same number, after brute force and the root sweep.

{% include figure.liquid path="assets/img/belief-propagation/bethe-counting.svg" alt="Variable and factor contributions being added while an edge-overlap contribution is subtracted" caption="The local normalizers count node and factor neighborhoods; each incidence overlap is counted twice and removed once. The identity is exact on a tree and defines the Bethe estimate on a loopy graph." %}

Why does a formula built from _normalized_ messages reproduce an unnormalized quantity? Attach an explicit scale to every normalization. In $Z^i$ each incoming factor-to-variable scale appears once; in $Z^a$ each incoming variable-to-factor scale appears once; and $Z^{ia}$ contains both scales belonging to incidence $(i,a)$. Every arbitrary constant therefore cancels from the ratio, leaving a gauge-invariant number. To see that this number is $Z$, prune a leaf variable and its adjacent factor: the ratio of the expression before and after pruning is precisely the branch contribution that pruning removed. Repeating until one node remains multiplies all removed contributions and terminates at the root normalization — the same induction as before, written multiplicatively.

The identity doubles as the best available unit test. On a tree, beliefs must normalize, each factor belief must marginalize to its adjacent variable belief, and the local-normalizer expression must reproduce brute-force $Z$ on small instances. When it does not, the cause is almost always an orientation error, a forgotten excluded neighbor, or inconsistent normalization — not a subtlety of the algorithm.

### BP as reparameterization

There is a third way to read what has just happened, and it turns out to be the most portable one. Substituting the beliefs back into the reconstruction formula gives

$$
P(\mathbf s)=\frac{\prod_a b_a(\mathbf s_{\partial a})}{\prod_i b_i(s_i)^{d_i-1}}
$$

on a tree. The right-hand side is built entirely from _local_ quantities, yet it reproduces the original global distribution exactly. BP has not changed the distribution at all; it has rewritten the same object in a new parameterization, one whose local factors happen to be the true marginals.

This is the reparameterization view, and it is useful because it survives partially into the loopy case: loopy BP fixed points also reparameterize the distribution, in the sense that the product form above leaves the joint distribution invariant, even though the local factors are then no longer the true marginals <d-cite key="wainwright2003tree"></d-cite>. The approximation is thus located precisely — not in the rewriting, which is exact, but in the identification of the new local factors with marginals.

### Implementing it without lying to yourself

<div class="bp-algorithm">
<strong>Reference implementation, in words</strong>
<ol>
<li>Store one message per <em>directed</em> edge. A common bug is storing one per undirected edge, which silently makes each message include the neighbor it is being sent to.</li>
<li>Work in logs; subtract the max before exponentiating inside every factor update.</li>
<li>Apply the same normalization to every message after every update, so that residuals are comparable across iterations.</li>
<li>On a tree, run leaves-inward then root-outward and stop. Iterating to a tolerance instead is wasteful and hides ordering bugs.</li>
<li>Assemble beliefs only from <em>incoming</em> messages, and remember that the belief at $i$ uses all of $\partial i$ while the message out of $i$ excludes one.</li>
</ol>
</div>

Three checks catch nearly every implementation error, and all three are cheap:

- **Normalization.** Every belief sums to one. Failures here are almost always a missing renormalization after an update.
- **Local consistency.** $\sum_{\mathbf s_{\partial a\setminus i}}b_a=b_i$ for every incidence. Failures indicate a transposed factor matrix or a wrong exclusion.
- **Exactness on a tree.** Beliefs and $\log Z$ match brute force on an instance small enough to enumerate. Failures that survive the first two checks are usually an incorrect elimination of the excluded neighbor in one of the two update types.

The running example in this chapter exists precisely so that these checks have something to compare against: $Z=108$, $b_1=(13/27,14/27)$, $b_2=(4/9,5/9)$. An implementation that reproduces those three values on this graph is almost certainly correct.

## The variational meaning of BP

The tree recursion explains the computation. It does not explain why the same equations keep appearing on graphs where the derivation is invalid. The variational view supplies that second reading: BP fixed points are stationary points of a particular approximate objective.

For any distribution $Q$ supported where the unnormalized weight $W$ is positive,

$$
\log Z
=\max_Q\left\{\mathbb E_Q[\log W(\mathbf s)]+H(Q)\right\},
$$

with equality at $Q=P$; this Gibbs variational principle is just the nonnegativity of $D_{\mathrm{KL}}(Q\Vert P)$ rearranged <d-cite key="wainwright2008graphical"></d-cite>. The maximization is over an intractable set, so approximations restrict the set, approximate the entropy, or both. Bethe does both.

### The tree entropy, and the leap

On a tree, a globally consistent distribution is fully reconstructed from its factor and variable marginals:

$$
Q(\mathbf s)
=
\frac{\prod_a b_a(\mathbf s_{\partial a})}
{\prod_i b_i(s_i)^{d_i-1}} .
$$

Taking $\mathbb E_Q[-\log Q]$ term by term gives the exact tree entropy

$$
H(Q)=\sum_a H(b_a)-\sum_i(d_i-1)H(b_i).
$$

Each factor contributes its own entropy; each variable is over-counted once per adjacent factor and corrected $d_i-1$ times. On a loopy graph the reconstruction formula is no longer valid — the right-hand side need not even be a probability distribution — but the entropy expression is adopted anyway and renamed the **Bethe entropy**. That substitution is the approximation. Everything after it is exact algebra applied to an inexact objective.

### Stationarity, with the multipliers written out

Collect the pieces into a dimensionless free energy:

$$
\mathcal F_{\mathrm B}(b)
=
\sum_a\sum_{\mathbf s_a}b_a(\mathbf s_a)
\log\frac{b_a(\mathbf s_a)}{f_a(\mathbf s_a)}
-\sum_i\sum_{s_i}b_i(s_i)\log g_i(s_i)
+\sum_i(1-d_i)\sum_{s_i}b_i(s_i)\log b_i(s_i).
$$

The first term carries the factor energy and factor entropy, the second the unary energy, and the third applies the counting number $1-d_i$ to the variable entropy only. The unary weight must be counted exactly once per variable: folding it into a $(1-d_i)\log(b_i/g_i)$ term would multiply $\log g_i$ by the wrong counting number and break $\mathcal F_{\mathrm B}=-\log Z$ on a tree. Equivalently, absorb each $g_i$ into an adjacent factor and drop the unary term.

Now minimize over beliefs that are nonnegative, normalized, and locally consistent. Attach multipliers $\alpha_a,\alpha_i$ to normalization and $\lambda_{ia}(s_i)$ to each consistency constraint. At an interior point, differentiating in $b_a$ gives

$$
\log b_a(\mathbf s_a)=\log f_a(\mathbf s_a)+\text{const}+\sum_{i\in\partial a}\lambda_{ia}(s_i),
\qquad\text{i.e.}\qquad
b_a\propto f_a\prod_{i\in\partial a}e^{\lambda_{ia}(s_i)} ,
$$

and differentiating in $b_i$ gives a matching exponential form in the multipliers arriving from all adjacent factors. The multipliers are redundant up to additive constants, exactly mirroring message gauge freedom. Identify $e^{\lambda_{ia}(s_i)}$ with a variable-to-factor message $\chi^{i\to a}_{s_i}$; imposing

$$
b_i(s_i)=\sum_{\mathbf s_{\partial a\setminus i}}b_a(\mathbf s_{\partial a})
$$

then forces the factor-to-variable summary to be proportional to $\sum_{\mathbf s_{\partial a\setminus i}}f_a\prod_{j\ne i}\chi^{j\to a}$, and substituting back reproduces the variable product update. The fixed-point equations are therefore not an analogy bolted onto the variational principle — they are a reparameterization of its first-order conditions <d-cite key="yedidia2005constructing,wainwright2003tree"></d-cite>.

<div class="bp-callout warning">
<strong>Assumptions matter.</strong> This derivation differentiates logarithms of beliefs, so it describes strictly positive interior points. Hard constraints push beliefs onto the boundary of the simplex, where the stationarity argument does not apply as written. Treat those by a limiting positive model, by constrained directional arguments, or by a theorem designed for boundary points — never by silently evaluating $\log 0$.
</div>

### Local consistency is weaker than global realizability

There is a second, easily missed approximation hiding in the constraint set. The beliefs range over the **local** marginal polytope: nonnegative, normalized, and consistent on each incidence. On a tree, every such collection is realized by an actual distribution, via the reconstruction formula. On a loopy graph it need not be.

The standard witness — the **frustrated triangle** — is a triangle of binary variables with

$$
b_{ij}(0,1)=b_{ij}(1,0)=\tfrac12,\qquad b_{ij}(0,0)=b_{ij}(1,1)=0
$$

on all three edges, and $b_i=(\tfrac12,\tfrac12)$ at every vertex. Each pair belief marginalizes correctly, so this point is locally consistent. Yet it asserts that all three pairs disagree simultaneously, which no assignment of three binary variables can achieve on an odd cycle. No global distribution has these marginals; the point lies strictly outside the marginal polytope.

So Bethe relaxes in two directions at once — an approximate entropy over an enlarged feasible set. Neither relaxation is a bug to be apologized for, but both must be stated when reporting results. Tree-reweighted methods change the counting numbers to obtain convexity and genuine upper bounds on $\log Z$ <d-cite key="wainwright2005new"></d-cite>; generalized BP enlarges the regions so short loops sit inside a single cluster <d-cite key="yedidia2005constructing"></d-cite>. Stable loopy-BP fixed points relate to local minima of the Bethe free energy under suitable conditions, but converses and global statements need care <d-cite key="heskes2003stable"></d-cite>.

## What loopy BP actually computes

Nothing stops us from iterating the same local equations on a graph with cycles. What disappears is the guarantee that incoming branches are disjoint: a message can travel around a loop and return, carrying a transformed version of information that originated at its own destination. The product update then treats correlated summaries as independent evidence.

That description is correct but vague. There is a sharp version. Run parallel BP for $t$ iterations and read off the belief at variable $i$. That belief is the _exact_ marginal of the root of a finite tree — the depth-$t$ **computation tree** at $i$, obtained by unrolling all nonbacktracking walks of length $t$ ending at $i$, with the model's factors copied along the way and the initial messages supplying the boundary condition <d-cite key="weiss2000correctness,tatikonda2002loopy"></d-cite>.

{% include figure.liquid path="assets/img/belief-propagation/computation-tree.svg" alt="A triangle beside its depth-three computation tree, in which each non-root node has a single child and variable one reappears at the leaves" caption="Unrolling a triangle. Loopy BP performs exact inference on this tree, not on the original graph. Variable 1 reappears at depth 3 as an independent copy of itself — that replication is the approximation." %}

This reframes the error precisely. Loopy BP is not performing approximate inference on the true graph; it is performing _exact_ inference on a different model — one in which a variable is replicated into many independent copies with the same local weights. BP is accurate exactly when that surrogate resembles the original, which is why correlation decay and loop length control quality, and why short loops hurt most.

It also explains why two kinds of error need not move together. The computation tree reproduces local statistics and global counting at different rates, so marginal accuracy and partition-function accuracy are separate questions rather than proxies for one another — and neither one dominates in general.

The triangle worked out in Part 2 makes this concrete, and cuts against the usual expectation. There the Bethe partition estimate is $2+\sqrt5$ against an exact $4$, a relative error of $5.9\%$, while the occupation is $\approx0.2764$ against an exact $1/4$, a relative error of $10.6\%$. The _local_ quantity is the less accurate one. Anyone reporting "the marginals looked reasonable, so $\log Z$ should be fine" — or the reverse — is asserting a correlation that this example already breaks.

Three consequences are worth extracting, because they convert vague intuitions into checkable statements.

**Initialization is a boundary condition, not a warm start.** The leaves of the depth-$t$ computation tree carry whatever messages the iteration began with. On a tree of finite depth this genuinely matters; it stops mattering only if the influence of the boundary decays with depth. So "BP converged to a different answer from a different start" is not a numerical accident — it means the surrogate model retains memory of its boundary, which is precisely the situation in which the approximation is unreliable.

**Correlation decay is the relevant hypothesis.** If the influence of the depth-$t$ boundary on the root decays as $t$ grows, the computation tree's root marginal stabilizes and stops depending on the initialization. That is the condition under which loopy BP is trustworthy, and it is why rigorous results in this area are usually theorems about correlation decay rather than about the algorithm <d-cite key="tatikonda2002loopy"></d-cite>.

**Girth controls how bad the surrogate is.** A vertex first reappears in its own computation tree at depth equal to the length of the shortest cycle through it. Long cycles postpone the replication; a triangle inflicts it almost immediately. This is the precise sense in which short loops are the enemy, and it also predicts the failure of naive intuition on graphs like grids, which are full of length-four cycles.

None of this makes loopy BP correct. It makes its incorrectness _specific_, which is what allows an application to argue that the specific error is tolerable.

## Living with loops

Three questions must stay separate, and conflating them is the most common way to misreport BP results.

1. **Does the iteration converge?** Parallel updates may oscillate or diverge. Damping, $m^{(t+1)}\leftarrow(1-\gamma)\widehat m^{(t+1)}+\gamma m^{(t)}$, and sequential or residual schedules change the numerical behavior <d-cite key="elidan2006residual"></d-cite>. None of them repairs the independence approximation; they change only whether the iteration settles.
2. **If it converges, are the beliefs accurate?** Convergence supplies a Bethe stationary point, not the true marginals. Early empirical studies made exactly this distinction, finding good accuracy in some regimes and confident, badly wrong beliefs in others <d-cite key="murphy1999loopy"></d-cite>.
3. **If the beliefs are accurate, is a decoded assignment optimal?** Marginal accuracy and MAP recovery are different objectives with different failure modes.

A practical stopping criterion is the residual

$$
r^{(t)}=\max_{e}\lVert m_e^{(t+1)}-m_e^{(t)}\rVert_1
$$

evaluated after fixing the gauge. Tolerance, schedule, damping, initialization, and iteration cap are part of the algorithm and belong in any report of results. Sufficient convergence conditions exist in terms of interaction strength and graph structure, but they are sufficient, not necessary — plenty of models converge comfortably outside them <d-cite key="mooij2007sufficient"></d-cite>.

Sparse random graphs are often described as locally tree-like. The safe statement is that in standard sparse ensembles a uniformly chosen **fixed-radius** neighborhood is a tree with probability tending to one as $N$ grows. That motivates cavity approximations; it does not make any finite loopy graph a tree, and it does not by itself imply weak long-range correlation. Rigorous asymptotic results need model-specific hypotheses <d-cite key="dembo2010ising"></d-cite>.

The cavity method is a language for the same edge-deletion construction: a cavity message describes the system with one interaction removed. Replica calculations go further, introducing replicated systems and analytic continuation in their number. They have produced remarkably accurate predictions, but the replica trick and the replica-symmetric ansatz are formal statistical-physics machinery unless paired with a separate rigorous theorem <d-cite key="mezard2009information,zdeborova2016statistical"></d-cite>. Loop-series expansions make the missing cyclic contributions explicit as a finite sum of generalized-loop corrections to the Bethe estimate <d-cite key="chertkov2006loop"></d-cite>, and graph-zeta identities give a complementary algebraic account <d-cite key="watanabe2009graph"></d-cite>. None of these changes the lesson: the tree proof tells us exactly which approximation loopy BP is making, and the computation tree shows us what it is making it on.

## What has actually been proved

<div class="bp-algorithm">
<strong>Tree sum–product, operationally</strong>
<ol>
<li>Choose any root.</li>
<li>Pass messages from the leaves toward the root using the two local recursions.</li>
<li>Read off the root marginal and $Z$.</li>
<li>Pass the complementary messages outward.</li>
<li>Assemble all variable and factor beliefs; optionally re-evaluate $Z$ through the local-normalizer identity as a check.</li>
</ol>
</div>

<table class="bp-decision-table">
<thead><tr><th>Claim</th><th>Finite tree</th><th>Loopy graph</th></tr></thead>
<tbody>
<tr><td>Messages are pinned subtree partition functions</td><td>Theorem</td><td>False in general</td></tr>
<tr><td>Two sweeps give all exact marginals</td><td>Theorem</td><td>No such guarantee</td></tr>
<tr><td>Local-normalizer formula equals $\log Z$</td><td>Identity</td><td>Approximation (Bethe)</td></tr>
<tr><td>Bethe entropy equals true entropy</td><td>Identity</td><td>Approximation</td></tr>
<tr><td>Local consistency implies a global distribution</td><td>Theorem</td><td>False in general</td></tr>
<tr><td>Interior fixed points are Bethe stationary points</td><td>Theorem</td><td>Theorem, under positivity and gauge conventions</td></tr>
<tr><td>The iteration converges</td><td>In two sweeps</td><td>Not guaranteed</td></tr>
<tr><td>Converged beliefs are exact</td><td>Yes</td><td>No</td></tr>
</tbody>
</table>

The right-hand column is not a list of defects. It is the specification of what an application must argue for itself. The exact tree computation gives a controlled template, the computation tree names precisely how that template is being reused, and every practical claim must say why the reuse is reasonable on its own graph.

### Two habits worth keeping

The first is to name the object before the algorithm. "We ran BP" is not a description of a computation; "we estimated the marginals of this factor graph by iterating sum–product with damping $0.5$ under a synchronous schedule to residual $10^{-10}$" is. Every ambiguity in the first phrasing corresponds to a decision that changes the answer.

The second is to keep a small exactly-solvable instance in the test suite forever. Trees are the ideal candidate because they turn approximation questions into equality questions: on a tree there is no tolerance to argue about, no schedule that changes the answer, and no ambiguity about what the right number is. When a large loopy run behaves strangely, the first question is always whether the same code still returns $108$ on four variables.

### Where this goes next

The rest of the series reuses this one construction in settings that look unrelated on the surface.

- **Optimization** replaces sums with minima and probabilities with costs, but keeps the cut-and-summarize structure exactly. That is Part 2.
- **Decoding** applies the same equations to a parity-check factor graph, where the loops are deliberately engineered to be long — a direct application of the girth observation above.
- **Survey propagation** changes what a message _is_, replacing a distribution over values with a distribution over cavity states, once the solution space stops being describable by a single one.
- **Neural message passing** keeps the aggregation pattern and learns the update, abandoning the probabilistic semantics in exchange for flexibility.
- **Tensor-network contraction** runs the same fixed-point iteration over tensors to approximate an environment, where the "messages" are boundary objects rather than beliefs.

In each case the questions from this chapter transfer unchanged: what global quantity is being computed, what a single message summarizes, where the factorization enters, what convergence means, and what can be checked independently.

Part 2 puts this to work. It turns costs into Boltzmann weights, takes the zero-temperature limit to get min-sum, and follows one sustained example — independent sets — from an exact tree recursion through a genuine phase transition to the point where the approximation visibly breaks.
