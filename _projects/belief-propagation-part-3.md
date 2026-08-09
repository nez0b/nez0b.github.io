---
layout: distill
title: Survey Propagation and the Shattered Solution Space
description: When one cavity marginal stops being enough — clustering, warnings, surveys over pure states, and the thresholds they separate
img: assets/img/belief-propagation/solution-space-clustering.png
permalink: /projects/belief-propagation/survey-propagation/
tags: belief-propagation survey-propagation satisfiability replica-symmetry-breaking cavity-method
importance: 99
category: work
show_on_projects: false
series: belief-propagation
series_part: 3
series_previous_url: /projects/belief-propagation/combinatorial-optimization/
series_previous_label: "Part 2"
series_next_url: /projects/belief-propagation/graph-neural-networks/
series_next_label: "Part 4"
status: draft
giscus_comments: false
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
bibliography: belief-propagation.bib
toc:
  - name: Satisfiability as a factor graph
  - name: Warnings are zero-temperature messages
  - name: When one marginal is not enough
  - name: A survey is a distribution over messages
  - name: Decimation, and what it does not prove
  - name: Four thresholds that are not the same threshold
  - name: What has actually been proved
---

{% include belief_propagation/series_nav.liquid %}

[Part 2]({{ '/projects/belief-propagation/combinatorial-optimization/' | relative_url }}) ended on an admission. When the solution space of a constraint-satisfaction problem fragments into many well-separated clusters, a single cavity marginal averages over clusters that no individual assignment can straddle — and the message itself becomes the wrong object, not merely a badly-estimated one.

This chapter takes that seriously. The route is: specialize belief propagation to satisfiability, take the zero-temperature limit already derived in Part 2 to obtain **warning propagation**, watch it fail for a structural rather than numerical reason, and then rebuild the message one level up as a **survey** — a probability distribution over the warnings a clause would send, one per cluster. That construction is survey propagation <d-cite key="mezard2002analytic,braunstein2005survey"></d-cite>.

It is also the chapter where the epistemic labels matter most. Much of what follows is a prediction of the one-step replica-symmetry-breaking (1RSB) cavity method, not a theorem. Some of it _is_ a theorem. The two are interleaved in the literature, and separating them is most of the work.

<div class="bp-callout warning">
<strong>Status of this chapter.</strong> The clustering and condensation thresholds below are 1RSB predictions. The satisfiability threshold is a theorem for large $k$. The success of survey-propagation-guided decimation near the threshold is an empirical finding, not a proof. Each claim below carries its label; none of them should be quoted without it.
</div>

## Satisfiability as a factor graph

A $k$-SAT formula over Boolean variables $x_1,\dots,x_n$ is a conjunction of clauses, each a disjunction of $k$ literals. Write it as a factor graph exactly as in Part 1: variables are circles, clauses are squares, and an edge $(i,a)$ exists when variable $i$ appears in clause $a$.

Each clause $a$ contributes a hard indicator factor,

$$
f_a(\mathbf x_{\partial a})=\mathbf 1\!\left[\text{clause } a \text{ is satisfied by } \mathbf x_{\partial a}\right],
$$

so the uniform measure over satisfying assignments is the factor-graph law of Part 1 with $g_i\equiv1$:

$$
P(\mathbf x)=\frac{1}{Z}\prod_{a}f_a(\mathbf x_{\partial a}),
\qquad
Z=\#\{\text{satisfying assignments}\}.
$$

Two pieces of notation make the SAT case readable. Each edge carries a sign: literal $x_i$ appears in clause $a$ either **positively** or **negatively**. For a variable $i$ and a clause $a\ni i$, split the _other_ clauses containing $i$ into

$$
\partial_a^{s}i=\{b\ne a:\ b \text{ agrees with } a \text{ on } i\},
\qquad
\partial_a^{u}i=\{b\ne a:\ b \text{ disagrees with } a \text{ on } i\},
$$

where "agrees" means $b$ is satisfied by the same value of $x_i$ that satisfies $a$. The $s$ and $u$ stand for _satisfying_ and _unsatisfying_ relative to $a$'s demand on $i$.

<div class="bp-callout derivation">
<strong>A tree formula you can enumerate.</strong>
Take
$$\varphi=(x_1\lor x_2\lor\lnot x_3)\land(x_3\lor x_4)\land(\lnot x_4\lor x_5).$$
Its factor graph has $5$ variable nodes and $3$ clause nodes joined by $7$ edges — $8$ nodes, $7$ edges, connected, therefore a tree. Everything Part 1 proved applies exactly.
Complete enumeration of all $2^5=32$ assignments finds <strong>13</strong> satisfying ones, and every variable takes both values somewhere in that set. The uniform-over-solutions marginals are
$$P(x_1{=}1)=\tfrac8{13},\quad P(x_2{=}1)=\tfrac8{13},\quad P(x_3{=}1)=\tfrac9{13},\quad P(x_4{=}1)=\tfrac7{13},\quad P(x_5{=}1)=\tfrac{10}{13}.$$
Belief propagation on this tree returns exactly these numbers. Keep them: they are the control case against which the failure below is measured.
</div>

## Warnings are zero-temperature messages

Part 2 derived min-sum by writing $m=\exp(-\beta M)$ and taking $\beta\to\infty$. For hard constraints the energy of a clause is $0$ when satisfied and $+\infty$ when violated, and the surviving information in a message collapses to a single bit.

Define the **warning** $u_{a\to i}\in\{0,1\}$:

$$
u_{a\to i}=1 \quad\Longleftrightarrow\quad
\text{clause } a \text{ forces } x_i \text{ to the value that satisfies } a.
$$

A clause forces its remaining variable exactly when every _other_ variable in it has already been pushed the wrong way by its own other clauses. That gives the warning-propagation recursion:

$$
\boxed{
u_{a\to i}=\prod_{j\in\partial a\setminus i} h_{j\to a},
\qquad
h_{j\to a}=\mathbf 1\!\left[\textstyle\sum_{b\in\partial_a^{u}j}u_{b\to j}\;>\;\sum_{b\in\partial_a^{s}j}u_{b\to j}\right].
}
$$

Read it in words: the local field $h_{j\to a}$ is $1$ when $j$'s other clauses push it _away_ from the value $a$ needs, strictly more than they push it toward that value. If every other variable in $a$ is pushed away, then $a$ has one option left and must issue a warning.

This is not a new algorithm either. It is min-sum from Part 2, restricted to a two-state message and written in the sign conventions of satisfiability <d-cite key="mezard2009information"></d-cite>.

<div class="bp-callout exact">
<strong>Exact on a tree.</strong> On a tree factor graph, warning propagation converges in one inward and one outward sweep, and a variable is <em>frozen</em> — taking the same value in every satisfying assignment — precisely when it receives a warning. The proof is Part 1's, with the pinned subtree partition function replaced by its zero-temperature limit.
</div>

<div class="bp-callout derivation">
<strong>Warnings on two tree formulas.</strong>
On $\varphi$ above, every variable takes both values across the $13$ solutions, so no variable is frozen and warning propagation returns $u_{a\to i}=0$ on every edge. That is the correct answer, and a deliberately boring one.
Now take the chain
$$\varphi_{\text{chain}}=(x_1)\land(\lnot x_1\lor x_2)\land(\lnot x_2\lor x_3),$$
which has exactly <strong>one</strong> satisfying assignment, $x=111$. Here every variable is frozen, warnings propagate down the chain, and the algorithm reports all three. Between these two extremes — nothing frozen, everything frozen — warning propagation is a complete description of a tree instance.
</div>

### From min-sum to a single bit, carefully

It is worth doing the collapse explicitly, because the step from "a message is a distribution" to "a message is one bit" is where SAT stops looking like Part 2.

Take a variable $j$ and write its cavity distribution toward clause $a$ in log-odds form, $h_{j\to a}=\log\frac{\chi^{j\to a}(\text{sat }a)}{\chi^{j\to a}(\text{unsat }a)}$. At inverse temperature $\beta$ with hard clauses, each incoming warning contributes an amount that diverges linearly in $\beta$, and everything else contributes $O(1)$. Writing $h=\beta H+O(1)$ and keeping only the leading term,

$$
H_{j\to a}=\sum_{b\in\partial_a^{s}j}u_{b\to j}\;-\;\sum_{b\in\partial_a^{u}j}u_{b\to j},
$$

a difference of integer counts. The sign of $H$ is all that survives: a strictly negative $H_{j\to a}$ means $j$ is being pushed away from what $a$ needs, which is the indicator in the boxed recursion above. Ties ($H=0$) mean $j$ is genuinely free, and the convention that $\mathbf 1[\cdot]$ uses a strict inequality is what encodes that.

This is the same collapse Part 2 performed for the hard-core model, where $\frac1\beta\log(1+e^{\beta U})\to\max(0,U)$ turned a soft competition into a hard one. Hard constraints simply push it further: the surviving object is not a real-valued score but a bit, because the only question a clause can ask is "am I down to my last option?"

<div class="bp-callout warning">
<strong>What the collapse costs.</strong> A bit can say "forced" or "not forced". It cannot say "forced in some clusters and not in others" — there is no room in $\{0,1\}$ for that distinction. Warning propagation therefore has no way to <em>represent</em> a shattered landscape, let alone converge on one. That representational limit, not a numerical one, is what the next section exhibits and what surveys repair.
</div>

## When one marginal is not enough

Now put the formula on a random graph and increase the clause density $\alpha=m/n$, the ratio of clauses to variables.

At low $\alpha$ the satisfying assignments form, in the 1RSB picture, a single connected blob: you can walk from any solution to any other by short sequences of single-variable flips that stay satisfying. A single cavity marginal describes this well, because there is one thing to describe.

Above a **clustering** (or dynamical) threshold $\alpha_d$, the picture changes qualitatively. The solution set shatters into exponentially many _clusters_, each internally connected but separated from the others by extensive Hamming distance — you cannot walk between clusters without passing through violated assignments <d-cite key="krzakala2007gibbs"></d-cite>. Write the number of clusters at free-energy density $f$ as

$$
\mathcal N(f)\;\simeq\;e^{\,n\Sigma(f)},
$$

where $\Sigma$ is the **complexity**. The support of $\Sigma\ge0$ is the range of cluster types that exist; where $\Sigma$ first hits zero at the dominant $f$, only sub-exponentially many clusters carry the measure — the **condensation** threshold $\alpha_c$.

{% include figure.liquid path="assets/img/belief-propagation/solution-space-clustering.svg" alt="Left: solutions forming one connected set joined by single-flip moves. Right: the same solutions split into six separated clusters" caption="Schematic of the shattering transition. Below $\alpha_d$ single-flip moves connect the solution set and one cavity marginal describes it; above $\alpha_d$ the set splits into clusters separated by extensive Hamming distance, and a single marginal averages over states that no individual assignment realizes. The picture illustrates the 1RSB prediction — it is not itself evidence for it." %}

Here is why this breaks the message, not merely its accuracy. A BP cavity marginal $\chi^{i\to a}(x_i)$ is a single distribution. In a shattered landscape, the honest answer to "what is $x_i$?" is _cluster-dependent_: frozen to $1$ in some clusters, frozen to $0$ in others, free in a third group. Averaging those into one number produces something near $1/2$ that describes no cluster at all — and, worse, the fixed-point equations that produced it assumed a single pure state to begin with.

<div class="bp-callout derivation">
<strong>A shattered instance small enough to check by hand.</strong>
Two independent exclusive-or gadgets on four variables:
$$\varphi'=(x_1\lor x_2)\land(\lnot x_1\lor\lnot x_2)\land(x_3\lor x_4)\land(\lnot x_3\lor\lnot x_4).$$
Enumeration gives exactly <strong>4</strong> satisfying assignments: $0101$, $0110$, $1001$, $1010$. Now count single-variable flips that stay satisfying: there are <em>none</em>. Every solution has zero satisfying neighbours at Hamming distance $1$, and the pairwise distances between solutions are all $2$ or $4$. Under the standard flip-connectivity definition, this instance has <strong>4 clusters, each a single isolated point</strong>.
By symmetry every variable is $1$ in exactly two of the four solutions, so the true marginal is $P(x_i{=}1)=1/2$ for all $i$ — a number that is simultaneously correct on average and useless for constructing a solution, since rounding all four variables to their marginals gives no valid assignment. This is a pedagogical gadget, not a random instance; $\alpha_d$ and $\alpha_c$ are asymptotic statements about typical large formulas. But it makes "a solution can be isolated from every other solution" a checkable fact rather than an assertion.
</div>

### What belief propagation actually returns there

The gadget is small enough to run sum–product on directly, so we need not speculate about what BP would say.

Running ordinary BP on the uniform-over-solutions measure of $\varphi'$ converges immediately to

$$
b_i(x_i{=}1)=\tfrac12 \qquad\text{for all four variables},
$$

which is _exactly_ the true marginal — each variable is $1$ in two of the four solutions. BP is not wrong here. It is answering the question it was asked, correctly, and the answer is useless.

That is the distinction this chapter turns on. In Part 1, loopy BP was inaccurate: it computed exact marginals of the wrong model. Here BP is _accurate_ and still unhelpful, because the marginal itself does not carry the information needed to build a solution. Rounding each variable to its most likely value is a coin flip per variable, and only $4$ of the $16$ roundings are satisfying. The failure is in the choice of summary statistic, not in its estimation.

<div class="bp-callout warning">
<strong>Two different failures, two different fixes.</strong> If the estimate is wrong, better scheduling, damping or a tighter approximation may help. If the estimated <em>quantity</em> is wrong, none of them will — you need a different message. Part 2's failure modes were all of the first kind. This one is of the second.
</div>

## A survey is a distribution over messages

The fix is to stop asking for a marginal and start asking for a distribution over cluster-conditional answers.

Fix a cluster. Inside it, warning propagation is well defined and returns a warning $u_{a\to i}\in\{0,1\}$ on every edge. Different clusters return different warnings. So define the **survey**

$$
Q_{a\to i}(u)\;=\;\Pr\!\left[\text{a randomly chosen cluster sends warning } u \text{ on edge } a\to i\right],
$$

a probability distribution over $\{0,1\}$ rather than a bit. This is the object one level up: BP's messages are distributions over _variable values_; surveys are distributions over _messages_. That hierarchy is exactly what the 1RSB cavity method formalizes <d-cite key="mezard2001bethe"></d-cite>.

For $k$-SAT the surveys can be reduced to a single number per edge, $\eta_{a\to i}=Q_{a\to i}(1)$, the probability that clause $a$ warns $i$. The update, in the notation established above, is

$$
\boxed{
\eta_{a\to i}
=\prod_{j\in\partial a\setminus i}
\frac{\Pi^{u}_{j\to a}}
{\Pi^{u}_{j\to a}+\Pi^{s}_{j\to a}+\Pi^{0}_{j\to a}},
}
$$

with the three cavity weights

$$
\begin{aligned}
\Pi^{u}_{j\to a}&=\Big[1-\prod_{b\in\partial_a^{u}j}(1-\eta_{b\to j})\Big]\prod_{b\in\partial_a^{s}j}(1-\eta_{b\to j}),\\
\Pi^{s}_{j\to a}&=\Big[1-\prod_{b\in\partial_a^{s}j}(1-\eta_{b\to j})\Big]\prod_{b\in\partial_a^{u}j}(1-\eta_{b\to j}),\\
\Pi^{0}_{j\to a}&=\prod_{b\in\partial_a^{u}j}(1-\eta_{b\to j})\prod_{b\in\partial_a^{s}j}(1-\eta_{b\to j}).
\end{aligned}
$$

The three cases are: $j$ is pushed away from what $a$ wants ($u$), pushed toward it ($s$), or pushed neither way ($0$). Clause $a$ warns $i$ only when every other variable is in the first case.

### The joker, and why it is not a third truth value

The third state deserves its own paragraph, because it is the piece most often misread. Alongside "must be true" and "must be false", survey propagation carries a symbol usually written $\star$ — the **joker**, meaning _this variable is unconstrained within this cluster_. It is not a third logical value that variables can take; assignments remain Boolean throughout. It is a statement about a cluster: within this pure state, $x_i$ is free to flip.

Formally the joker is what makes $\Pi^0$ a separate case rather than folded into $\Pi^s$. Maneva, Mossel and Wainwright made this precise by constructing an explicit Markov random field over the enlarged alphabet $\{0,1,\star\}$ and proving that survey propagation is _ordinary belief propagation_ on that enlarged model <d-cite key="maneva2007new"></d-cite>.

<div class="bp-callout exact">
<strong>What is a theorem here.</strong> That SP equals BP on an enlarged $\{0,1,\star\}$ MRF is proved <d-cite key="maneva2007new"></d-cite>. That the marginals of that MRF correctly count clusters of the original formula is the 1RSB physics conjecture riding on top. The first statement licenses reusing every convergence and reparameterization tool from Parts 1–2; it does not license the second.
</div>

### Counting clusters: the complexity

Surveys buy more than a per-variable answer. Because each cluster contributes its own Bethe free entropy, the 1RSB construction yields an estimate of how _many_ clusters there are at each free-energy density — the complexity $\Sigma(f)$ introduced above.

Operationally, one introduces a Parisi parameter $y$ conjugate to the cluster free energy and computes a generalized partition function over clusters,

$$
\mathcal Z(y)=\sum_{\text{clusters }\gamma} e^{-y\,n f_\gamma}\;\simeq\;\int \mathrm df\; e^{\,n[\Sigma(f)-yf]},
$$

whose saddle point ties $y$ to a particular cluster free energy through $\Sigma'(f)=y$. The complexity is then recovered by a Legendre transform of $\frac1n\log\mathcal Z(y)$.

Two special values matter. At $y=0$ every cluster is weighted equally, so the calculation counts clusters rather than weighting them by size — this is survey propagation as normally run, and it is why SP answers "in what fraction of clusters is $x_i$ frozen to $1$?" rather than "in what fraction of _solutions_". At the $y$ that maximizes $\Sigma-yf$ one recovers the thermodynamically dominant clusters instead. The distinction is easy to lose and changes what the output means.

<div class="bp-callout heuristic">
<strong>Status.</strong> The complexity, its Legendre structure, and the identification of $\Sigma=0$ with condensation are 1RSB predictions. They are internally consistent, they match numerics well, and for random $k$-SAT at large $k$ the resulting satisfiability threshold was later proved correct <d-cite key="ding2022satisfiability"></d-cite>. None of that makes the intermediate objects theorems.
</div>

Survey propagation as usually implemented is the $y\to0$ member of a one-parameter family, where $y$ is a reweighting (Parisi) parameter conjugate to cluster free energy. At the other end of the family the equations degenerate to ordinary BP. So SP is not a different algorithm bolted on beside BP — it is BP evaluated on a richer message space, with a parameter that interpolates back to the original <d-cite key="mezard2002random,maneva2007new"></d-cite>.

### Reading the update as three exclusive cases

The survey update looks forbidding until one notices it is just a normalized three-way case split, written once per neighbour.

Fix a clause $a$ and a variable $j\in\partial a$, and ask what the _rest_ of the formula does to $j$ in a randomly chosen cluster. Exactly one of three things happens.

<table class="bp-notation">
<thead><tr><th>Case</th><th>Meaning</th><th>Weight</th></tr></thead>
<tbody>
<tr><td>$u$</td><td>$j$'s other clauses push it <em>away</em> from what $a$ needs</td><td>$\Pi^{u}_{j\to a}$</td></tr>
<tr><td>$s$</td><td>$j$'s other clauses push it <em>toward</em> what $a$ needs</td><td>$\Pi^{s}_{j\to a}$</td></tr>
<tr><td>$0$</td><td>$j$ is unconstrained — the joker</td><td>$\Pi^{0}_{j\to a}$</td></tr>
</tbody>
</table>

Each weight is built the same way: "at least one clause on the relevant side warns, and none on the other side does." The bracket $\bigl[1-\prod(1-\eta)\bigr]$ is precisely "at least one warning arrives," and the bare product $\prod(1-\eta)$ is "none arrives." A configuration in which both sides warn is a contradiction and is excluded — which is why the three weights need not sum to one before normalizing.

Clause $a$ then warns $i$ only when _every_ other variable is in case $u$, giving the boxed product. Set every $\eta$ to $0$ or $1$ and the whole thing collapses back to warning propagation; that limit is a useful implementation check.

<div class="bp-callout derivation">
<strong>Cost per sweep.</strong> Each directed message costs $O(k)$ for a clause of arity $k$, since the three weights are products over $\partial a\setminus i$ that can be computed once per variable and reused. A full sweep is $O(\sum_a k_a)=O(m k)$ — linear in the formula size, exactly like BP. The extra power of SP does not come from more computation per edge; it comes from carrying a distribution over warnings rather than a warning.
</div>

## Decimation, and what it does not prove

Surveys are not assignments. The standard way to convert them is **SP-guided decimation**, the same pattern as the BP-guided decimation of Part 2 with a different message underneath.

<div class="bp-algorithm">
<strong>SP-guided decimation</strong>
<ol>
<li>Run SP to a fixed point on the current formula.</li>
<li>From the converged surveys compute, for each free variable, its bias — the imbalance between the probability of being frozen to $1$ and to $0$ across clusters.</li>
<li>Fix the most biased variable to its preferred value.</li>
<li>Simplify the formula (remove satisfied clauses, shorten the rest) and return to step 1.</li>
<li>When the surveys become trivial — all $\eta\approx0$, meaning no clause constrains anything — stop and finish with a local search such as WalkSAT.</li>
</ol>
</div>

Empirically this works remarkably far. On large random 3-SAT it finds solutions at clause densities close to the satisfiability threshold $\alpha_s\approx4.267$, a regime where every previously known method stalls <d-cite key="mezard2002analytic,braunstein2005survey"></d-cite>.

<div class="bp-callout warning">
<strong>What that sentence is not.</strong> "SP solves random 3-SAT near the threshold" is an empirical report about a family of random instances at particular sizes, not a theorem, and not a statement about worst-case SAT. There is no proof that SP-guided decimation succeeds up to $\alpha_s$. Analyses of the simpler BP-guided decimation show that its success is itself governed by a tree recursion whose stability must be checked rather than assumed <d-cite key="montanari2007solving,riccitersenghi2009cavity"></d-cite>, and decimation of either kind can walk into an unsatisfiable residual formula by freezing variables inconsistently.
</div>

The failure mode is worth naming precisely, because it is the same shape as Part 2's decimation caveat. Each fixing step is irreversible and is made on the basis of a fixed point computed _before_ that step. If an early, confidently-biased variable is set wrongly, nothing later revisits it, and the residual formula can become unsatisfiable while every individual step looked well-justified. Backtracking variants exist and improve the reachable density, at the cost of no longer being a single forward pass.

### Where the variational picture went

Part 1 gave BP a second reading: fixed points are stationary points of the Bethe free energy, and that variational view explained why the same equations keep appearing on graphs where the tree derivation is invalid. It is fair to ask what happened to that reading here.

It survives, one level up. The 1RSB construction has its own variational object — a free energy over _distributions of messages_ rather than over messages — and survey propagation's fixed points are its stationary points, exactly as BP's fixed points were stationary points of Bethe <d-cite key="mezard2001bethe"></d-cite>. The Parisi parameter $y$ enters as the conjugate variable that selects which clusters dominate, and the complexity $\Sigma$ is the Legendre transform that counts them.

This is the sense in which SP is not an ad-hoc patch. It is the same variational machinery applied to a richer space of order parameters, and the hierarchy continues: two-step replica-symmetry breaking would carry distributions over distributions over messages, and so on. For random $k$-SAT, one step is believed to suffice near the threshold; for other models it demonstrably does not.

The Maneva–Mossel–Wainwright result gives the cleanest statement of what this buys, and it is worth repeating precisely because it is the rigorous anchor of the chapter: SP _is_ belief propagation, on a specific enlarged Markov random field over $\{0,1,\star\}$ <d-cite key="maneva2007new"></d-cite>. Every tool from Parts 1 and 2 — reparameterization, convergence analysis, scheduling, damping — applies to it unchanged, because it is BP. What is conjectural is not the algorithm but the _interpretation_: that the enlarged model's marginals describe clusters of the original formula.

<div class="bp-callout exact">
<strong>The one-sentence version.</strong> Survey propagation is belief propagation on a larger alphabet. The mathematics of running it is Part 1's. The physics of why that larger alphabet is the right one is a prediction — well-supported, partially confirmed, and still a prediction.
</div>

## Four thresholds that are not the same threshold

Part 2 insisted on threshold discipline for the hard-core model. Random $k$-SAT needs it more, because at least four distinct thresholds appear in the same discussions and are routinely conflated.

<table class="bp-decision-table">
<thead><tr><th>Threshold</th><th>What it marks</th><th>Status</th></tr></thead>
<tbody>
<tr><td>$\alpha_{\mathrm{alg}}$</td><td>where the best known polynomial-time algorithm still succeeds w.h.p.</td><td>Rigorous achievability results exist <d-cite key="cojaoghlan2010better"></d-cite>; it is a statement about known algorithms, not a hardness barrier</td></tr>
<tr><td>$\alpha_d$</td><td>solution space shatters into exponentially many clusters</td><td>1RSB prediction <d-cite key="krzakala2007gibbs"></d-cite></td></tr>
<tr><td>$\alpha_c$</td><td>sub-exponentially many clusters carry the measure</td><td>1RSB prediction <d-cite key="krzakala2007gibbs"></d-cite></td></tr>
<tr><td>$\alpha_s$</td><td>satisfiable below, unsatisfiable above</td><td><strong>Theorem</strong> for all large $k$ <d-cite key="ding2022satisfiability"></d-cite>; a sharp threshold is known to <em>exist</em> for every $k$ <d-cite key="friedgut1999sharp"></d-cite></td></tr>
</tbody>
</table>

For large $k$ these are ordered

$$
\alpha_{\mathrm{alg}}\;\ll\;\alpha_d\;<\;\alpha_c\;<\;\alpha_s,
$$

and the first inequality is the uncomfortable one. Known polynomial-time algorithms stall at roughly $2^k\ln k/k$ clauses per variable <d-cite key="cojaoghlan2010better"></d-cite>, while satisfiability persists to roughly $2^k\ln 2$ <d-cite key="achlioptas2004threshold,ding2022satisfiability"></d-cite> — an exponentially large gap in which solutions provably exist and no efficient algorithm is known to find them. Clustering begins somewhere above where algorithms already stopped, so "clustering causes algorithmic failure" is, at best, an incomplete story <d-cite key="zdeborova2016statistical"></d-cite>.

The satisfiability threshold is the one genuine theorem in this list, and its history is instructive: existence of a sharp threshold came first without its location <d-cite key="friedgut1999sharp"></d-cite>, then increasingly tight bounds <d-cite key="achlioptas2004threshold"></d-cite>, and finally a proof for large $k$ whose value matches the 1RSB prediction made two decades earlier <d-cite key="mezard2002analytic,ding2022satisfiability"></d-cite>. That agreement is a genuine scientific success for the cavity method. It is not a licence to treat its other predictions as proved.

### The gap that clustering does not explain

The ordering $\alpha_{\mathrm{alg}}\ll\alpha_d$ deserves more than a line, because it undercuts the tidiest version of the story.

The appealing narrative runs: solutions shatter at $\alpha_d$, shattering traps local algorithms, therefore algorithms fail above $\alpha_d$. The numbers refuse to cooperate. For large $k$, known polynomial-time algorithms stall around $2^k\ln k/k$, while clustering sets in at a density that is larger by a factor growing with $k$ <d-cite key="cojaoghlan2010better,zdeborova2016statistical"></d-cite>. There is an exponentially wide band of densities in which the solution space is, by the 1RSB account, still one connected blob — and no efficient algorithm is known to find a point in it.

So clustering cannot be _the_ cause of algorithmic hardness, because hardness arrives first. What clustering plausibly explains is why certain _specific_ local strategies fail, and why SP outperforms BP where it does. The broader question of why any polynomial-time algorithm should stop where it does remains open, and is one of the more interesting open problems the cavity method has surfaced without solving.

This is worth stating plainly because the opposite claim is common and comfortable. A reader who takes away only "clustering makes SAT hard" has acquired a memorable sentence that the numbers do not support.

### When to reach for a survey

<table class="bp-decision-table">
<thead><tr><th>Situation</th><th>Reach for</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Tree or near-tree factor graph</td><td>BP / warning propagation</td><td>Exact; a survey adds nothing to describe one cluster</td></tr>
<tr><td>Loopy graph, BP converges, marginals look decisive</td><td>BP with decimation</td><td>A single pure state is plausibly being described</td></tr>
<tr><td>BP converges but every marginal sits near $1/2$</td><td>Suspect shattering</td><td>The gadget above is the minimal instance of this symptom</td></tr>
<tr><td>BP oscillates on a random CSP near threshold</td><td>Survey propagation</td><td>Non-convergence and clustering are frequently the same fact</td></tr>
<tr><td>Structured/industrial instance, not a random ensemble</td><td>A real SAT solver</td><td>SP's evidence base is random ensembles; CDCL dominates elsewhere</td></tr>
</tbody>
</table>

The last row matters more than its length suggests. Everything in this chapter is about _random_ formulas near a threshold. Modern conflict-driven clause-learning solvers routinely dispatch structured industrial instances with millions of variables that no message-passing method would touch, and SP is not competitive there. The regime where SP is remarkable — large random instances close to $\alpha_s$ — is exactly the regime where CDCL struggles. They are complementary tools aimed at different distributions, and the honest comparison names the distribution first.

## What has actually been proved

<table class="bp-decision-table">
<thead><tr><th>Claim</th><th>Status</th></tr></thead>
<tbody>
<tr><td>Warning propagation is the $\beta\to\infty$ limit of BP for hard constraints</td><td>Exact identity</td></tr>
<tr><td>WP is exact on a tree formula; warnings identify frozen variables</td><td>Theorem (Part 1's argument)</td></tr>
<tr><td>SP is BP on an enlarged $\{0,1,\star\}$ MRF</td><td>Theorem <d-cite key="maneva2007new"></d-cite></td></tr>
<tr><td>Clusters exist above $\alpha_d$; complexity $\Sigma(f)$ counts them</td><td>1RSB prediction</td></tr>
<tr><td>SP marginals describe per-cluster freezing</td><td>1RSB prediction</td></tr>
<tr><td>SP-guided decimation solves random 3-SAT near $\alpha_s$</td><td>Empirical</td></tr>
<tr><td>$\alpha_s(k)$ for large $k$</td><td>Theorem <d-cite key="ding2022satisfiability"></d-cite></td></tr>
</tbody>
</table>

### A note on what SP does not do

Three clarifications, because each is a common over-reading.

**SP does not sample uniformly from solutions.** At $y=0$ it weights clusters equally, so a cluster containing one solution counts as much as a cluster containing $2^{0.01n}$ of them. If the goal is uniform sampling, that bias is a defect rather than a feature, and it must be corrected for explicitly.

**SP does not certify unsatisfiability.** If it converges to trivial surveys, the correct reading is "this method found no constraint structure," not "no solution exists." Refutation is a genuinely different problem, and message passing has nothing to say about it.

**SP is not a general-purpose replacement for BP.** Below $\alpha_d$ the surveys concentrate on a single warning per edge and SP reduces to warning propagation with extra bookkeeping. Running it everywhere costs constant-factor time for no gain, and the extra machinery obscures what is happening.

The chapter's structural lesson generalizes past SAT. When a method fails, there are two very different diagnoses available: the estimate is poor, or the estimated object is wrong. Parts 1 and 2 dealt with the first — loopy BP computing exact marginals of the wrong model, the computation tree. This chapter dealt with the second. No amount of damping, scheduling or reinitialization repairs a single cavity marginal in a shattered landscape, because the quantity being computed does not summarize the thing being asked about. The repair was to enlarge the message.

That is a move worth keeping. Whenever a message-passing scheme fails stubbornly, it is worth asking whether the message has enough room to carry the answer — before concluding that message passing was the wrong idea.

[Part 4]({{ '/projects/belief-propagation/graph-neural-networks/' | relative_url }}) turns to a family of methods that make the opposite trade: they keep the local aggregation pattern and discard the probabilistic content entirely, learning the update from data instead of deriving it from a distribution.
