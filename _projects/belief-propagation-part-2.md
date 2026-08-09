---
layout: distill
title: Belief Propagation for Combinatorial Optimization
description: From Boltzmann weights to min-sum, independent sets, matching, colouring, phase transitions, and the limits of fixed-point iteration
img: assets/img/belief-propagation/hard-core-tree-transition.png
permalink: /projects/belief-propagation/combinatorial-optimization/
tags: belief-propagation combinatorial-optimization min-sum independent-set graph-colouring
importance: 99
category: work
show_on_projects: false
series: belief-propagation
series_part: 2
series_previous_url: /projects/belief-propagation/from-first-principles/
series_previous_label: "Part 1"
series_next_url: /projects/belief-propagation/survey-propagation/
series_next_label: "Part 3"
status: draft
giscus_comments: false
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: belief-propagation.bib
toc:
  - name: Optimization as a Boltzmann problem
  - name: From sum-product to min-sum
  - name: Independent set as a factor graph
  - name: A phase transition from one recursion
  - name: Matching changes the factorization
  - name: Colouring and local stability
  - name: Three ways BP can fail
  - name: When should you use BP
---

{% include belief_propagation/series_nav.liquid %}

Many combinatorial problems are global questions assembled from local rules. A matching may use at most one edge incident to each vertex. A proper colouring forbids equal colors across an edge. An independent set forbids simultaneous occupation of adjacent vertices. Each rule involves a handful of variables; the difficulty is that satisfying one rule changes which choices remain available everywhere else.

[Part 1]({{ '/projects/belief-propagation/from-first-principles/' | relative_url }}) derived belief propagation as exact dynamic programming on a tree, and identified precisely what breaks on a loopy graph. This chapter puts that machinery to work on optimization. The route is: turn costs into Boltzmann weights, let temperature interpolate between counting and optimizing, take the zero-temperature limit to obtain min-sum, and then follow one example — independent sets — from an exact tree recursion, through a genuine phase transition, to the point where the approximation visibly fails. Matching and colouring appear afterwards to show how the same framework responds when the constraint structure changes shape.

Every numerical claim below is either derived in closed form, checked against complete enumeration of a small instance, or — where the instance is too large for either, as in the spin-glass convergence experiment — reported explicitly as the output of one reproducible simulation rather than as a general fact.

<div class="bp-callout warning">
<strong>Scope.</strong> BP is not a general polynomial-time solver for NP-hard problems. Local factorization makes each update cheap; it does not guarantee that a loopy fixed point exists, is unique, reproduces the true model, or decodes a global optimum. Every positive claim in this chapter is attached to a stated graph class and objective.
</div>

## Optimization as a Boltzmann problem

Let $\mathbf x=(x_1,\ldots,x_n)$ be discrete decision variables whose cost decomposes into local terms,

$$
E(\mathbf x)=\sum_{i\in V}E_i(x_i)+\sum_{a\in F}E_a(\mathbf x_{\partial a}),
$$

where $E_i$ is a per-variable cost and $E_a$ couples the variables in $\partial a$. These play the roles that $g_i$ and $f_a$ played in Part 1, now written additively.

At inverse temperature $\beta\ge0$ define

$$
P_\beta(\mathbf x)=\frac{1}{Z(\beta)}e^{-\beta E(\mathbf x)},
\qquad
Z(\beta)=\sum_{\mathbf x}e^{-\beta E(\mathbf x)} .
$$

Because $e^{-\beta E}$ factorizes over the same groups as $E$, this is a factor graph with $f_a=e^{-\beta E_a}$, and all of Part 1 applies unchanged.

One model now supports several genuinely different questions:

<table class="bp-decision-table">
<thead><tr><th>Task</th><th>Quantity</th><th>Typical BP form</th></tr></thead>
<tbody>
<tr><td>Marginal inference</td><td>$P_\beta(x_i)$</td><td>sum–product beliefs</td></tr>
<tr><td>Counting / free energy</td><td>$\log Z(\beta)$</td><td>Bethe local-normalizer formula</td></tr>
<tr><td>Sampling</td><td>draws from $P_\beta$</td><td>BP-guided decimation</td></tr>
<tr><td>MAP</td><td>$\arg\max_\mathbf{x}P_\beta$</td><td>max-product</td></tr>
<tr><td>Optimization</td><td>$\min_\mathbf{x}E$</td><td>min-sum</td></tr>
</tbody>
</table>

MAP and energy minimization share an $\arg\max$ for every $\beta>0$, so the temperature is a free dial for that task alone. Finite-temperature marginals, by contrast, describe the whole ensemble of near-optimal states — which is more information, and sometimes the information you actually want.

The bridge between the two regimes is

$$
-\frac{1}{\beta}\log Z(\beta)\;\longrightarrow\;\min_{\mathbf x}E(\mathbf x)
\qquad(\beta\to\infty).
$$

The next term records degeneracy: if $N_\star$ assignments achieve the minimum $E_\star$, then

$$
Z(\beta)=e^{-\beta E_\star}\!\left(N_\star+o(1)\right),
$$

so the ground-state count appears as a subleading constant. This is a mathematical bridge, not an algorithm — evaluating $Z$ exactly can remain exponentially hard at every temperature.

<div class="bp-callout derivation">
<strong>Temperature, made concrete.</strong>
The five-vertex tree used throughout the next section has independence polynomial
$$Z(\lambda)=1+5\lambda+6\lambda^2+2\lambda^3 ,$$
one term per independent-set size. Every question above is a different way of reading this single polynomial. At $\lambda=1$ it counts: $Z(1)=14$. As $\lambda\to\infty$ the cubic term dominates, so $Z(\lambda)/\lambda^3\to2$ — recovering both the maximum size $3$ and the number of maximum sets, $N_\star=2$. At small $\lambda$ the linear term dominates and the model barely occupies anything.
The temperature is therefore not a numerical trick. It is a dial that selects which coefficient of the same polynomial dominates, and $\beta\to\infty$ is the statement that the top coefficient wins.
</div>

Hard constraints enter by setting $E_a=+\infty$ on forbidden local configurations, equivalently by a zero-valued factor. Two consequences follow immediately. Zeros sit on the boundary of the probability simplex, so the interior stationarity argument from Part 1 does not apply verbatim. And a zero can propagate: a message that becomes identically zero signals local inconsistency rather than a numerical problem, and implementations must decide deliberately whether to treat that as infeasibility or to regularize.

## From sum-product to min-sum

Write a positive message in exponential form,

$$
m_{i\to a}(x_i)=\exp\!\left[-\beta M_{i\to a}(x_i)+c_{i\to a}\right],
$$

where $c_{i\to a}$ absorbs normalization. The factor-to-variable sum–product update then contains

$$
\sum_{\mathbf x_{\partial a\setminus i}}
\exp\left\{-\beta\left[E_a(\mathbf x_{\partial a})+
\sum_{j\in\partial a\setminus i}M_{j\to a}(x_j)\right]\right\}.
$$

Everything now hinges on how a $\beta$-weighted sum behaves as $\beta$ grows. For any finite set $\mathcal Y$,

$$
\min_y A(y)-\frac{\log|\mathcal Y|}{\beta}
\;\le\;
-\frac1\beta\log\sum_y e^{-\beta A(y)}
\;\le\;
\min_y A(y).
$$

The upper bound keeps only the smallest term in the sum; the lower bound replaces every term by that smallest one. So the soft minimum always sits at or below the hard minimum, and the gap is at most $\log\lvert\mathcal Y\rvert/\beta$, closing as $\beta\to\infty$. Applying this inside the update gives

$$
\boxed{
M_{a\to i}(x_i)
=
\min_{\mathbf x_{\partial a\setminus i}}
\left[E_a(\mathbf x_{\partial a})+
\sum_{j\in\partial a\setminus i}M_{j\to a}(x_j)\right]+C_{a\to i},
}
$$

with the variable update inheriting the product-to-sum conversion directly:

$$
\boxed{
M_{i\to a}(x_i)
=
E_i(x_i)+\sum_{b\in\partial i\setminus a}M_{b\to i}(x_i)+C_{i\to a}.
}
$$

This is **min-sum**; flipping signs gives max-sum/max-product. The constants $C$ form an additive gauge — adding a constant to every state of one message shifts all downstream values equally and cannot change an $\arg\min$. Implementations usually subtract $\min_x M(x)$ after each update, which keeps numbers bounded and makes two runs comparable.

Read structurally, min-sum is the shortest-path recursion. On a chain it _is_ the Viterbi algorithm: $M$ plays the role of the cost-to-go, the factor update is the relaxation step over the previous stage, and the additive gauge is the usual freedom to shift potentials.

<div class="bp-callout exact">
<strong>Values are not solutions.</strong> Min-sum values give optimal costs conditioned on each boundary state. To recover an optimizer, store the minimizing inner assignment in each factor update, choose an optimal root state, and backtrack through the stored choices. This is textbook dynamic programming: the inward pass computes value functions, the backward pass reconstructs a witness. Skipping the bookkeeping and decoding state-by-state from the values alone is valid only when every local minimizer is unique.
</div>

That last caveat is not hypothetical, as the next section demonstrates with an explicit tie.

On a loopy graph the order of limits starts to matter: taking $\beta\to\infty$ before iterating, during iterating, or after convergence can give different numerical behavior when optima are degenerate.

### Turning beliefs into an assignment

Beliefs are not solutions — and on a loopy graph they are beliefs, not marginals, since Part 1 showed the two coincide only on a tree. There is also no backtracking theorem to fall back on. Three strategies are common, and they differ in how much they admit to being heuristics.

**Independent rounding.** Take $\arg\max_{x_i}b_i(x_i)$ at every variable at once. This is the cheapest option and the most dangerous: nothing couples the rounding decisions, so the result can violate constraints that every individual belief respected. The zero-temperature independent-set example below produces exactly this hazard.

**Decimation.** Run BP, fix the single most polarized variable to its preferred value, condition the model on that choice, and re-run on the reduced problem. Each round is cheap, constraints stay satisfied by construction, and the beliefs are recomputed in light of every commitment made so far. The cost is $O(n)$ BP runs and a new failure mode: an early confident-but-wrong decision is never revisited, and the reduced problem can become unsatisfiable.

**Reinforcement.** Rather than freezing variables, feed a slowly growing external field back into each variable proportional to its own current belief. This nudges the whole system toward a self-consistent assignment without hard commitments, at the cost of another schedule to tune.

All three modify the algorithm. None inherits the tree guarantee, and all should be reported as part of the method rather than as post-processing.

## Independent set as a factor graph

An independent set $S\subseteq V$ contains no two adjacent vertices. Let $x_i\in\{0,1\}$ indicate $i\in S$. At fugacity $\lambda>0$ the hard-core model is

$$
P(\mathbf x)=\frac1Z\,\lambda^{\sum_i x_i}\prod_{(i,j)\in E}(1-x_ix_j),
\qquad
Z=\sum_{S\text{ independent}}\lambda^{|S|}.
$$

Larger $\lambda$ favors larger sets; the edge factor vanishes exactly on the forbidden state $x_i=x_j=1$. At $\lambda=1$ the partition function simply counts independent sets, which makes small instances easy to check by hand.

{% include figure.liquid path="assets/img/belief-propagation/independent-set-factor-graph.svg" alt="An independent-set problem graph beside its factor graph, with occupied nonadjacent vertices highlighted and one hard factor for each edge" caption="The independent-set factorization. Variable weights reward occupation by $\lambda$; every red edge factor removes the forbidden state $(x_i,x_j)=(1,1)$." %}

### The cavity recursion, derived

For a directed graph edge $i\to j$, let

$$
p^{i\to j}=\Pr\!\left(x_i=1\mid\text{constraint }(i,j)\text{ removed}\right)
$$

under the normalized cavity message. Condition on the two states of $x_i$. If $x_i=1$, every remaining neighbor $k\in\partial i\setminus j$ must be unoccupied, contributing $1-p^{k\to i}$ each. If $x_i=0$, each neighbor is unconstrained and its normalized message contributes $1$. Hence the cavity weights are

$$
w_1=\lambda\prod_{k\in\partial i\setminus j}\left(1-p^{k\to i}\right),
\qquad
w_0=1,
$$

and normalizing gives

$$
\boxed{
p^{i\to j}
=
\frac{\lambda\prod_{k\in\partial i\setminus j}(1-p^{k\to i})}
{1+\lambda\prod_{k\in\partial i\setminus j}(1-p^{k\to i})} .
}
$$

Once all incoming messages are available, the node occupation uses the full neighborhood:

$$
\rho_i=\frac{\lambda\prod_{k\in\partial i}(1-p^{k\to i})}
{1+\lambda\prod_{k\in\partial i}(1-p^{k\to i})} .
$$

This $\rho_i$ is exactly Part 1's belief evaluated at the occupied state, $\rho_i=b_i(1)$; a binary alphabet lets a single number stand for the whole distribution, which is why the notation collapses to a scalar here.

This is not a new algorithm. It is the generic factor update of Part 1, simplified by the fact that the variables are binary and the constraint is hard.

<div class="bp-callout derivation">
<strong>A tree you can count by hand.</strong>
Take five vertices with edges $(1,2),(2,3),(2,4),(3,5)$ — a hub at vertex $2$ with a tail through $3$ to $5$ — and set $\lambda=1$. Complete enumeration finds
$$Z=14\quad\text{independent sets.}$$
Running the recursion to convergence gives exact rational cavity values, for example $p^{1\to2}=\tfrac12$ and $p^{2\to1}=\tfrac14$, and node occupations
$$\rho_1=\tfrac37,\quad\rho_2=\tfrac17,\quad\rho_3=\tfrac27,\quad\rho_4=\tfrac37,\quad\rho_5=\tfrac5{14}.$$
Every value matches brute-force enumeration exactly. The hub is heavily suppressed ($\tfrac17$) because occupying it excludes three neighbors at once.
</div>

### Odds, log-domain, and the zero-temperature limit

The multiplicative structure is clearest in odds. With $r^{i\to j}=p^{i\to j}/(1-p^{i\to j})$, the identity $1-p^{k\to i}=1/(1+r^{k\to i})$ turns the recursion into

$$
r^{i\to j}=\lambda\prod_{k\in\partial i\setminus j}\frac{1}{1+r^{k\to i}},
$$

and with $h=\log r$,

$$
h^{i\to j}=\log\lambda-\sum_{k\in\partial i\setminus j}\log\!\left(1+e^{h^{k\to i}}\right).
$$

Now attach weights: put $\lambda=e^{\beta w}$ and scale $h=\beta U+o(\beta)$. Since $\frac1\beta\log(1+e^{\beta U})\to\max(0,U)$, the limiting recursion is

$$
\boxed{
U^{i\to j}=w-\sum_{k\in\partial i\setminus j}\max\!\left(0,U^{k\to i}\right).
}
$$

Each neighbor that would rather be occupied ($U^{k\to i}>0$) charges its surplus against occupying $i$; neighbors that prefer to stay empty charge nothing. This is the max-weight independent set recursion, and it is exact on a tree.

<div class="bp-callout derivation">
<strong>The same tree at zero temperature — including a tie.</strong>
With unit weights on the five-vertex tree, enumeration gives a maximum independent set of size $3$, attained by <em>two</em> different sets: $\{1,4,5\}$ and $\{1,3,4\}$.
The converged cavity scores give node scores
$$U_1=1,\quad U_2=-1,\quad U_3=0,\quad U_4=1,\quad U_5=0 .$$
Decoding by sign alone commits only to $\{1,4\}$ and leaves vertices $3$ and $5$ at exactly zero. The zeros are not a bug: they are the algorithm correctly reporting that the two optima disagree on those vertices. Backtracking resolves the tie consistently; independent per-node rounding could pick $3$ and $5$ together and produce an infeasible set.
</div>

### Where the approximation becomes visible

Now close the tree into a triangle and keep $\lambda=1$. There are exactly $4$ independent sets: the empty set and three singletons. The symmetric cavity equation $p=(1-p)/\bigl(2-p\bigr)$ has the closed-form solution

$$
p^\star=\frac{3-\sqrt5}{2}\approx0.381966 ,
$$

from which the Bethe estimate works out to the golden-ratio cube

$$
Z_{\mathrm B}=\varphi^3=2+\sqrt5\approx4.2361,
\qquad\text{against the exact } Z=4,
$$

and the occupation is $\rho_{\mathrm{BP}}=(3-\sqrt5)/(5-\sqrt5)\approx0.2764$ against the exact $1/4$.

BP converges here without difficulty. The error is not a convergence failure — it is the computation tree from Part 1 doing its job faithfully on the wrong model, treating the returning copy of each vertex as an independent one.

{% include figure.liquid path="assets/img/belief-propagation/exact-vs-bethe-loop-error.svg" alt="Bethe log-partition error for a hard-core model on a four-node path and a triangle over a range of fugacities" caption="Complete finite enumeration provides the reference. The tree here is a four-node path — a different, even smaller instance than the five-vertex tree above — chosen so both graphs stay enumerable across the whole fugacity range. The tree estimate is exact to numerical precision; the triangle fixed point converges, yet its Bethe partition estimate misses a loop-dependent term that grows with $\lambda$." %}

Notice also the direction of the error: the Bethe estimate _overcounts_. That is characteristic rather than universal, and it is exactly the sort of statement that needs a loop-correction analysis rather than a plot to justify in general <d-cite key="chertkov2006loop"></d-cite>.

Finally, the Bethe free-entropy density on a $d$-regular tree at symmetric cavity value $p$ is

$$
\phi_{\mathrm B}
=\log\!\left[1+\lambda(1-p)^d\right]-\frac d2\log\!\left(1-p^2\right),
$$

the first term being the variable normalizer $Z^i$ and the second removing the $d/2$ edge overlaps per site. This is the object whose branches we compare next.

## A phase transition from one recursion

On the infinite $d$-regular tree, impose a translation-invariant message $p^{i\to j}=p$. The recursion collapses to a single scalar map:

$$
p=f(p)=\frac{\lambda(1-p)^{d-1}}{1+\lambda(1-p)^{d-1}} .
$$

Since $f$ is strictly decreasing on $[0,1]$ and maps the interval into itself, it has exactly one fixed point $p^\star$. Uniqueness of the symmetric solution, however, says nothing about whether iteration converges to it — that is a question about $\lvert f^{\prime}\rvert$.

Differentiating,

$$
f^{\prime}(p)=-\frac{(d-1)\lambda(1-p)^{d-2}}{\left[1+\lambda(1-p)^{d-1}\right]^2} .
$$

At a fixed point the defining equation supplies two substitutions,

$$
\lambda(1-p)^{d-1}=\frac{p}{1-p},
\qquad
1+\lambda(1-p)^{d-1}=\frac{1}{1-p},
$$

and inserting them collapses the derivative to

$$
\boxed{f^{\prime}(p^\star)=-(d-1)\,p^\star .}
$$

The negative sign is the antiferromagnetic character of the constraint: more occupation on one side means less on the next. Because $f$ is decreasing, the natural instability is period-two — a perturbation that alternates between the two sublattices of the tree. Such a perturbation is amplified when $\lvert f^{\prime}(p^\star)\rvert>1$, i.e. when $(d-1)p^\star>1$. At threshold $p^\star=1/(d-1)$, and substituting back into the fixed-point equation gives

$$
\boxed{\lambda_c(d)=\frac{(d-1)^{d-1}}{(d-2)^d},\qquad d\ge3 .}
$$

For $d=3$ this is $\lambda_c=4$ exactly.

{% include figure.liquid path="assets/img/belief-propagation/hard-core-tree-transition.svg" alt="Hard-core occupation and two-sublattice order together with the derivative magnitude of the symmetric recursion, crossing one at fugacity four" caption="The infinite 3-regular-tree recursion. Above $\lambda_c=4$ an alternating two-sublattice solution appears and the symmetric recursion becomes locally unstable. The occupation curve is the symmetric branch throughout, including where that branch is no longer the stable one." %}

The right panel plots $\lvert f^{\prime}(p^\star)\rvert=(d-1)p^\star$ crossing one at exactly $\lambda=4$, confirming the algebra numerically. The left panel adds the order parameter $\lvert p_A-p_B\rvert$, where $p_A=f(p_B)$ and $p_B=f(p_A)$: identically zero below threshold, continuously growing above it.

### The physics statement and the algorithmic statement

These are usually quoted together and are not the same claim.

The **physics** statement concerns the measure: above $\lambda_c$ the hard-core Gibbs measure on the infinite regular tree stops being unique, and boundary conditions at infinity leave a trace at the root.

The **algorithmic** statement concerns the iteration: the symmetric fixed point still exists above $\lambda_c$ — the scalar equation $p=f(p)$ still has exactly one solution — but $\lvert f^{\prime}(p^\star)\rvert>1$ means synchronous iteration no longer converges to it. In exact arithmetic from a symmetric start the iteration would sit there indefinitely; perturbed by anything at all, including rounding, it departs and alternates between the two sublattice values.

So the same threshold marks a change in the model and a change in the behavior of a particular numerical scheme, and only the first is intrinsic. This is failure mode 1 of the next section, arriving with a closed-form threshold attached — and a reminder that non-convergence sometimes carries information about the model rather than merely signalling a tuning problem.

<div class="bp-callout warning">
<strong>Threshold discipline.</strong> $\lambda_c$ is the uniqueness and local-stability threshold for the hard-core measure on the <em>regular tree</em> under the translation-invariant recursion. It is genuinely load-bearing — approximate counting is possible up to it <d-cite key="weitz2006counting"></d-cite> and becomes NP-hard beyond it for general graphs of that degree <d-cite key="sly2010computational"></d-cite>. That pair of results is unusually sharp and unusually specific. It does not license relabelling $\lambda_c$ as the reconstruction threshold, the condensation threshold, the point where BP stops converging on a particular finite graph, or a universal easy/hard boundary for every hard-core instance.
</div>

One more caution about transferring the picture. The alternating solution lives naturally on a bipartite structure, and the infinite regular tree is bipartite. A finite non-bipartite graph — an odd cycle, say — cannot realize a clean two-sublattice phase at all; it is frustrated instead. The tree calculation predicts the onset of a symmetry-breaking instability, not the phase diagram of whatever finite graph you happen to hold.

## Matching changes the factorization

Independent set puts variables on vertices and constraints on edges. Matching does the opposite, and the contrast is instructive: the same algorithm behaves quite differently when the roles are exchanged.

Let $x_e\in\{0,1\}$ indicate whether edge $e$ is selected, with weight $w_e$. Every original vertex $v$ becomes a factor enforcing

$$
\sum_{e\ni v}x_e\le1,
\qquad
P(\mathbf x)\propto\prod_e e^{\beta w_e x_e}\prod_v\mathbf 1\!\left[\sum_{e\ni v}x_e\le1\right].
$$

{% include figure.liquid path="assets/img/belief-propagation/matching-duality.svg" alt="A triangle matching factor graph in which original edges are circular variables and original vertices are square at-most-one constraints" caption="Matching uses the dual factorization: each original edge becomes a binary variable and each original vertex becomes an at-most-one factor." %}

Note the arity: a vertex of degree $d$ produces a factor over $d$ variables. A naive factor update would cost $2^d$. The at-most-one structure rescues it — the constraint is a _cardinality_ constraint, and its message can be computed in $O(d)$ by tracking only the two relevant cases. That is worth stating explicitly, because it is the general lesson for high-arity factors: exploit the structure of the constraint or pay exponentially in its degree.

### Ratio messages, derived rather than sketched

Work with odds so that normalizations cancel. Remove edge-variable $e=(u,v)$ from vertex factor $v$, and let $r_{f\to v}$ denote the incoming occupied-to-unoccupied weight ratio for another incident edge $f$. Measure everything in units of the all-zero state.

If $x_e=1$, then every $f\in\partial v\setminus e$ must be zero, so exactly one configuration survives and the contribution is $1$. If $x_e=0$, the surviving configurations are "no other edge selected" (contributing $1$) plus "exactly one $f$ selected" (contributing $r_{f\to v}$ for each $f$). Therefore

$$
\boxed{
a_{v\to e}\equiv\frac{m_{v\to e}(1)}{m_{v\to e}(0)}
=\frac{1}{1+\sum_{f\in\partial v\setminus e}r_{f\to v}} .
}
$$

The edge variable simply relays its own weight together with the cavity from its far endpoint,

$$
r_{e\to v}=e^{\beta w_e}\,a_{u\to e},
$$

and combining both endpoints gives the occupation odds and marginal

$$
r_e=e^{\beta w_e}\,a_{u\to e}\,a_{v\to e},
\qquad
\Pr(x_e=1)=\frac{r_e}{1+r_e} .
$$

<div class="bp-callout derivation">
<strong>Checked against enumeration.</strong>
On the path $u-v-w-z$ with weights $w_{uv}=3$, $w_{vw}=2$, $w_{wz}=4$, the maximum-weight matching is $\{uv,wz\}$ with total weight $7$. At $\beta=1$ the equations above return edge marginals $0.9466$, $0.0063$, $0.9759$, matching complete enumeration to $2\times10^{-16}$. Raising $\beta$ to $6$ drives them to $1,0,1$ to four decimal places — the optimal matching. The factor graph is a tree, so this agreement is a theorem, not a coincidence.
</div>

Taking logarithms in the zero-temperature limit turns $a_{v\to e}$ into an additive **availability** score (hence the letter): how much weight the endpoint would forfeit by committing to this edge. That is the same quantity affinity-propagation-style algorithms pass around, and the reason max-product for matching has an unusually strong theory: on bipartite graphs, with the LP relaxation having a unique optimum, max-product converges to the maximum-weight matching, with the proof running through LP duality <d-cite key="bayati2008maxproduct"></d-cite>.

That theorem is valuable precisely because it is exceptional. It buys its guarantee with three specific hypotheses — bipartite structure, a linear objective whose relaxation is integral, and uniqueness — none of which generic loopy max-product enjoys.

It is worth being explicit about why the LP appears at all. Maximum-weight matching has a natural linear-programming relaxation: replace $x_e\in\{0,1\}$ by $x_e\in[0,1]$, keep the degree constraints, and maximize $\sum_e w_ex_e$. On a bipartite graph the constraint matrix is totally unimodular, so the relaxation has an integral optimum and the LP value equals the combinatorial one. Max-product's fixed points, read through the reparameterization lens of Part 1, encode a decomposition of the objective that certifies exactly this LP optimum — which is why the guarantee tracks LP integrality rather than any property of the message dynamics.

On a non-bipartite graph the same LP has fractional vertices — the half-integral solution assigning $1/2$ to every edge of an odd cycle is the standard example, and it is the matching analogue of the frustrated triangle from Part 1. Precisely there, the correspondence breaks and max-product loses its guarantee. That is a satisfying state of affairs: the algorithm fails exactly where the relaxation it is implicitly solving stops being tight.

## Colouring and local stability

A proper $q$-colouring assigns $s_i\in\{1,\ldots,q\}$ with $s_i\ne s_j$ on every edge. Soften it into an antiferromagnetic Potts factor,

$$
f_{ij}(s_i,s_j)=e^{-\beta\mathbf 1[s_i=s_j]},
$$

which forbids monochromatic edges only in the limit $\beta\to\infty$.

Because every factor has arity two, it is conventional here to merge each factor into its edge and keep one message per directed graph edge rather than the two alternating families of Part 1. Below, $\chi^{k\to j}$ is that single merged message — the composition of Part 1's $\chi$ and $\psi$ across one edge — which is why no $\psi$ appears in this section.

The factor update simplifies beautifully. With $\theta=1-e^{-\beta}$ and a normalized incoming cavity distribution $\chi^{k\to j}$,

$$
\sum_{s_k}f_{kj}(s_k,c)\,\chi^{k\to j}_{s_k}
=\underbrace{\sum_{s_k}\chi^{k\to j}_{s_k}}_{=1}-\left(1-e^{-\beta}\right)\chi^{k\to j}_c
=1-\theta\chi^{k\to j}_c ,
$$

so the whole algorithm reduces to one equation over colour distributions:

$$
\boxed{
\chi^{j\to i}_c\;\propto\;\prod_{k\in\partial j\setminus i}\left(1-\theta\chi^{k\to j}_c\right).
}
$$

Colour symmetry means the uniform message $\chi_c=1/q$ is always a fixed point. The interesting question is whether it is stable — whether an infinitesimal colour bias grows or decays as it propagates.

### Linearizing

Write $\chi_c=1/q+\epsilon_c$ with $\sum_c\epsilon_c=0$. For a single incoming neighbor,

$$
1-\theta\chi_c=\left(1-\frac\theta q\right)\left[1-\frac{\theta}{1-\theta/q}\,\epsilon_c+O(\epsilon^2)\right].
$$

Multiplying over the incoming neighbors and renormalizing kills the colour-independent prefactor, and because $\sum_c\epsilon_c=0$ the normalizer contributes nothing at first order. The outgoing perturbation is therefore

$$
\boxed{
\epsilon^{j\to i}_c\approx-\frac{\theta}{q-\theta}\sum_{k\in\partial j\setminus i}\epsilon^{k\to j}_c .
}
$$

At zero temperature $\theta\to1$ and the per-edge channel eigenvalue is $-1/(q-1)$. The negative sign is the antiferromagnetic response: raising the probability of colour $c$ at a neighbor suppresses it here. Perturbations compound along nonbacktracking walks, so on a graph with branching factor $B$ the natural first-moment criterion compares $B$ against $\lvert\lambda_2\rvert^{-1}$.

The classical reconstruction criterion is a second-moment statement rather than a first-moment one, since it tracks the variance of a propagated signal:

$$
B\,|\lambda_2|^2=1 .
$$

With $\lvert\lambda_2\rvert=1/(q-1)$ this gives $B=(q-1)^2$ — the Kesten–Stigum line for colouring. For sparse Erdős–Rényi graphs of average degree $c$ the branching factor is $c$, so the criterion reads $c_{\mathrm{KS}}=(q-1)^2$.

<div class="bp-callout heuristic">
<strong>What this criterion is, and is not.</strong> It diagnoses growth of infinitesimal second-moment perturbations of the uniform fixed point under a specified tree/ensemble model <d-cite key="kesten1966additional"></d-cite>. It is not automatically the colourability threshold, the clustering or condensation threshold, or a universal boundary between tractable and intractable instances; those separate and only sometimes coincide, and the relationships are model dependent <d-cite key="zdeborova2016statistical"></d-cite>.
</div>

### One arithmetic check settles the point

It would be easy to read $c_{\mathrm{KS}}=(q-1)^2$ as "the colouring threshold". A first-moment bound refutes that in three lines. The expected number of proper $q$-colourings of $G(n,c/n)$ is

$$
\mathbb E[\#\text{colourings}]=q^n\left(1-\frac1q\right)^{cn/2},
$$

treating each of the $\approx cn/2$ edges as improper with probability $1/q$ independently of the colouring. That independence is not exact at finite $n$ — for small graphs the true expectation differs — but it is correct to leading exponential order in the sparse $n\to\infty$ limit, which is the only regime the bound is used in. This expectation decays exponentially — so colourings almost surely do not exist — once

$$
\log q+\frac c2\log\!\left(1-\frac1q\right)<0
\qquad\Longleftrightarrow\qquad
c>c_{\mathrm{ann}}=\frac{-2\log q}{\log\!\left(1-\tfrac1q\right)} .
$$

Comparing the two expressions:

<table class="bp-data-table">
<thead><tr><th>$q$</th><th>$c_{\mathrm{KS}}=(q-1)^2$</th><th>$c_{\mathrm{ann}}$ (first-moment upper bound)</th><th></th></tr></thead>
<tbody>
<tr><td>3</td><td>4.00</td><td>5.42</td><td>KS below</td></tr>
<tr><td>4</td><td>9.00</td><td>9.64</td><td>KS below</td></tr>
<tr><td>5</td><td>16.00</td><td>14.43</td><td><strong>KS above</strong></td></tr>
<tr><td>6</td><td>25.00</td><td>19.65</td><td><strong>KS above</strong></td></tr>
<tr><td>8</td><td>49.00</td><td>31.15</td><td><strong>KS above</strong></td></tr>
</tbody>
</table>

From $q=5$ onward the Kesten–Stigum line sits _above_ a rigorous upper bound on colourability. At $c=16$ with $q=5$, proper colourings have already ceased to exist with high probability, so nothing about the stability of the uniform BP fixed point at that density can be the colourability threshold. The two quantities are simply different objects, and for $q\ge5$ they are not even close.

This is the cleanest available illustration of the discipline this series keeps insisting on. A linear-stability calculation answers a linear-stability question. Turning it into a statement about where solutions exist, or about where algorithms succeed, requires separate arguments — and here, an elementary one shows that the naive identification is false.

## Three ways BP can fail

"BP failed" conflates mechanisms with different diagnoses and different fixes.

### 1. Stationary but unstable

Below the Bethe critical temperature of a ferromagnet, the symmetric message remains an exact fixed point. Initialized exactly there, BP stays forever, even though an arbitrarily small perturbation grows away toward an ordered state. Algebraic fixed-point existence is strictly weaker than dynamical stability, and a solver that reports "converged" from a symmetric start has told you nothing about which phase the model is in.

### 2. Stable but not globally best

The Bethe objective is generally nonconvex, so it can have several local minima, each a stable BP fixed point with its own basin of attraction. Which one you reach is determined by initialization.

The figure below shows a tilted 3-regular ferromagnetic Ising reduction at $\beta J=0.8$, $\beta h=0.06$. Three fixed points exist. Two are stable, with linearized update derivatives $0.43$ and $0.61$; the third, at magnetization $-0.302$, has derivative $1.30$ and is unstable. The two stable ones are _not_ equivalent: their restricted Bethe objectives are $-1.2702$ and $-1.1550$. BP started aligned with the field reaches magnetization $+0.970$ and the lower value; started anti-aligned it settles at $-0.945$ and the higher one — a stable, converged, perfectly self-consistent, metastable answer.

{% include figure.liquid path="assets/img/belief-propagation/bethe-landscape-and-flow.svg" alt="A restricted tilted Ising Bethe objective with two stable and one unstable fixed point, beside BP trajectories ending at different magnetizations" caption="A uniform-message reduction of a 3-regular ferromagnetic Ising model. Field-aligned and anti-aligned initializations converge to different stable fixed points; one is metastable. The left curve is a one-dimensional restriction of the objective, not the full Bethe variational domain." %}

Two caveats belong with this figure. The curve is a one-dimensional slice through the uniform-message family, not the full higher-dimensional Bethe domain, so it should be read as an existence demonstration rather than a complete landscape. And stable fixed points correspond to local Bethe minima only under suitable regularity conditions <d-cite key="heskes2003stable"></d-cite>.

### 3. No convergence at all

Parallel updates can oscillate indefinitely. Frustration is the usual cause: when competing constraints cannot be satisfied simultaneously, no self-consistent set of messages exists nearby, and the iteration chases its own tail.

Take a random $3$-regular graph on $60$ spins with couplings $J_{ij}=\pm1$ drawn uniformly, at $\beta\lvert J\rvert=2$, updated synchronously. The residual does not decay at all — it saturates near $1$, the maximum possible change for a normalized binary message, and stays there for three thousand sweeps.

{% include figure.liquid path="assets/img/belief-propagation/spin-glass-nonconvergence.svg" alt="Residual versus sweep on log scale for five damping levels, all remaining far above the convergence tolerance after three thousand sweeps" caption="Parallel BP on a frustrated spin glass. Increasing damping lowers the residual by nearly two orders of magnitude but never reaches the tolerance; the iteration keeps moving indefinitely." %}

Damping helps, monotonically and unmistakably — and still does not converge:

<table class="bp-data-table">
<thead><tr><th>Damping $\gamma$</th><th>Final residual after 3000 sweeps</th><th>Below $10^{-10}$?</th></tr></thead>
<tbody>
<tr><td>$0$</td><td>$0.999$</td><td>no</td></tr>
<tr><td>$0.5$</td><td>$0.254$</td><td>no</td></tr>
<tr><td>$0.8$</td><td>$0.061$</td><td>no</td></tr>
<tr><td>$0.9$</td><td>$0.023$</td><td>no</td></tr>
<tr><td>$0.95$</td><td>$0.017$</td><td>no</td></tr>
</tbody>
</table>

This is one instance rather than a theorem about spin glasses, but it makes the mechanism concrete. Damping shrinks the step, so it shrinks the residual; it does not create a fixed point where none is being approached. Reading the last row as "nearly converged" would be a mistake — the residual is small because the updates are small, not because the messages have settled.

The honest options at this point are to change the schedule (sequential or residual-priority updates often succeed where synchronous ones fail <d-cite key="elidan2006residual"></d-cite>), to use continuation in $\beta$ or $\lambda$ from an easy regime, to switch to a method that provably descends an objective, or to conclude that the model is outside BP's useful range. What is not an option is quoting beliefs from a non-converged run.

### Keeping the four questions apart

<table class="bp-decision-table">
<thead><tr><th>Question</th><th>Diagnostic</th><th>Not implied by success</th></tr></thead>
<tbody>
<tr><td>Numerical convergence</td><td>gauge-fixed message residual below tolerance</td><td>accurate marginals</td></tr>
<tr><td>Local stability</td><td>spectral radius of the linearized update below one</td><td>global Bethe optimum</td></tr>
<tr><td>Approximation quality</td><td>enumeration on small instances, bounds, trusted asymptotics</td><td>an optimal decoded assignment</td></tr>
<tr><td>Optimization quality</td><td>primal objective value and constraint feasibility</td><td>a correct partition function</td></tr>
</tbody>
</table>

If a genuine Bethe optimum is the goal, one can minimize the objective directly with a convergent double-loop method instead of iterating the raw BP map <d-cite key="yuille2002cccp"></d-cite>. If a bound is the goal, tree-reweighted constructions give convex surrogates and true upper bounds on $\log Z$ <d-cite key="wainwright2005new"></d-cite>. Neither dissolves the original hardness; they change which approximation is being optimized and which guarantees come with it.

There is also a regime where the message itself is the wrong object. When the solution space of a random constraint-satisfaction problem shatters into exponentially many well-separated clusters, a single cavity marginal averages over clusters that no single assignment can straddle <d-cite key="krzakala2007gibbs"></d-cite>. Survey propagation responds by passing a distribution over cavity _states_ rather than over assignments <d-cite key="mezard2002analytic,braunstein2005survey"></d-cite>. That is a genuinely different algorithm resting on further statistical-physics structure, and it belongs to a later chapter rather than a footnote here.

## When should you use BP

<table class="bp-decision-table">
<thead><tr><th>Structure and goal</th><th>Expected behavior</th><th>What to check</th></tr></thead>
<tbody>
<tr><td>Tree or low-treewidth graph; exact marginals or MAP</td><td>Exact dynamic programming</td><td>both message directions; backtracking for ties</td></tr>
<tr><td>Sparse, weakly correlated loopy graph</td><td>Often fast and accurate</td><td>several initializations, residuals, small-instance enumeration</td></tr>
<tr><td>Strong frustration or long-range order</td><td>Multiple, oscillatory, or confidently wrong fixed points</td><td>stability, damping sensitivity, correlation diagnostics</td></tr>
<tr><td>Bipartite matching with an integral, unique LP optimum</td><td>Convergence and correctness guarantees available</td><td>the theorem's exact hypotheses</td></tr>
<tr><td>Hard optimization with no structural theorem</td><td>Heuristic candidate generator</td><td>feasibility and objective against baselines and bounds</td></tr>
<tr><td>Clustered random CSP regime</td><td>Plain BP may be the wrong object</td><td>ensemble assumptions; survey-type methods</td></tr>
</tbody>
</table>

### What each sweep actually costs

The three examples in this chapter have deceptively similar update equations and quite different cost profiles, which is worth making explicit because it is the practical reason one encoding is preferred over another.

- **Independent set.** Binary variables, arity-two hard factors. Each directed message is a single scalar $p^{i\to j}$, and one sweep costs $O(\lvert E\rvert)$ scalar operations. This is the cheap end, and it is why the hard-core model is such a convenient laboratory.
- **Colouring.** Alphabet size $q$, arity-two factors. A generic pair update would cost $O(q^2)$ per edge, but the antiferromagnetic structure collapses the sum to $1-\theta\chi_c$, giving $O(q)$ per directed edge and $O(q\lvert E\rvert)$ per sweep. The saving comes from the factor being a rank-one perturbation of a constant, not from anything about BP.
- **Matching.** Binary variables, but factor arity equals the vertex degree $d$. A generic factor update would cost $2^d$. The at-most-one constraint reduces it to $O(d)$ via the ratio form derived above, so a sweep is again $O(\lvert E\rvert)$.

The pattern generalizes: BP is linear in the number of edges only when each local update exploits the structure of its factor. High-arity factors are not automatically expensive, but they are automatically expensive if implemented naively — and a great deal of practical work on message passing is exactly this kind of per-factor algebra.

### What a reproducible BP experiment reports

<div class="bp-algorithm">
<strong>Reporting checklist</strong>
<ol>
<li><strong>Model.</strong> The factorization actually used, including how hard constraints are encoded and whether any regularization was applied to zeros.</li>
<li><strong>Messages.</strong> Parameterization (probabilities, odds, log-odds, energies) and the gauge in which residuals are measured.</li>
<li><strong>Dynamics.</strong> Schedule (synchronous, sequential, residual-priority), damping, initialization, tolerance, and iteration cap — plus how many runs converged, not just the ones that did.</li>
<li><strong>Decoding.</strong> Rounding, decimation, or reinforcement, with its own schedule, and whether the output was checked for feasibility.</li>
<li><strong>Reference.</strong> An exact comparison wherever the instance is small enough to enumerate, and an explicit statement when none is available.</li>
<li><strong>Scope.</strong> Which numbers are finite-instance observations and which are infinite-tree or ensemble statements. These should never appear in the same sentence without a label.</li>
</ol>
</div>

The last item is the one most often skipped, and this chapter has tried to model the alternative. The number $\lambda_c=4$ is a theorem about an infinite regular tree. The number $2+\sqrt5$ is an exact Bethe prediction for one triangle. The number $14$ is an exact count for one five-vertex tree. The residual $0.017$ is one spin-glass instance at one damping setting. Those are four different kinds of claim, and collapsing them into "BP gives about the right answer" would discard the only thing that makes any of them useful.

The unifying idea is the one from Part 1, now under load. BP replaces a component by a boundary-conditioned summary. On a tree that summary is complete, and the results here are theorems: exact counts on the five-vertex tree, exact matching marginals on the path, exact max-weight recursions. On a loopy graph the same summary encodes a hypothesis about which correlations may be neglected — a hypothesis that the triangle refutes numerically, that the hard-core threshold locates precisely, and that the Bethe landscape shows can be satisfied by more than one self-consistent answer at once.

That is why the algorithm rewards being derived rather than memorized. Once you know it computes exact marginals on the computation tree, every question about its behavior on your problem becomes a concrete question about how much your graph resembles that tree.

### The five questions, answered for this chapter

The series opened with five questions to ask of any message-passing method. Optimization answers them as follows.

1. **What global quantity is being computed?** A minimum of $E$, or the marginals of $P_\beta$ near that minimum — and, through $\log Z(\beta)$, the count of near-optimal states. These are different targets served by the same equations at different temperatures.
2. **What does one message summarize?** Whatever the encoding makes local: an occupation probability $p^{i\to j}$ for independent set, an availability $a_{v\to e}$ for matching, a colour distribution $\chi^{j\to i}$ for colouring — and at zero temperature, a cost difference $U^{i\to j}$ rather than a probability.
3. **Where does factorization enter?** In the assumption that a variable's cavity neighbors are independent once it is pinned. Exact on the trees here; refuted quantitatively by the triangle.
4. **What does convergence mean?** Only that the residual stopped changing. The spin-glass run shows an iteration that never converges; the hard-core threshold shows one that stops converging for a reason intrinsic to the model; the tilted Ising shows two different converged answers from two different starts.
5. **What is independently checkable?** Every tree result here, against enumeration. Every threshold, against its own defining fixed-point equation. And on any loopy instance small enough to enumerate, the approximation error itself — which is the number worth reporting.
