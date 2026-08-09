---
layout: distill
title: How to Optimize Statevector Simulation
description: Insert one zero bit to enumerate every amplitude pair exactly once—and understand where the speedup goes
img: assets/img/quantum-simulation/bit-insertion.png
permalink: /projects/quantum-simulation/how-to-optimize-statevector-simulation/
tags: quantum-computing statevector bit-manipulation performance
importance: 99
category: work
show_on_projects: false
series: quantum-simulation
series_part: 1
status: draft
giscus_comments: false
authors:
  - name: PoJen Wang
    url: "https://nez0b.github.io"
    affiliations:
      name: Independent research project
toc:
  - name: A gate becomes an indexing problem
  - name: The amplitude-pair invariant
  - name: Insert a zero bit
  - name: Visit every pair exactly once
  - name: What the benchmark measures
  - name: The bandwidth wall
  - name: Where to optimize next
---

{% include quantum_simulation/series_nav.liquid %}

A statevector simulator can be introduced in one line: store the $2^N$ amplitudes of an
$N$-qubit state and apply matrices to them. A literal implementation of that sentence is
also a good way to make the simulator unnecessarily slow.

The important observation is local. A one-qubit gate never needs an arbitrary pair of
amplitudes. For target qubit $q$, it mixes amplitudes whose basis-state indices differ
**only** at bit $q$. Once those pairs can be generated directly, the gate becomes a
single in-place sweep with no full operator, no permutation, and no output statevector.

This chapter develops that indexing trick. A
[Julia teaching notebook](https://github.com/kshyatt/QNumerics_2026_SV/blob/main/state_vector.jl)
provides the pedagogical starting point, while the measurements come from my Rust
implementation, [rustatevec](https://github.com/nez0b/rustatevec). The point is not the
syntax of either language. It is the bit algebra that tells both implementations exactly
which data to touch.

## A gate becomes an indexing problem

Write the state as

$$
\lvert\psi\rangle=\sum_{x=0}^{2^N-1}\psi_x\lvert x\rangle.
$$

Let a one-qubit gate be

$$
G=\begin{pmatrix}g_{00}&g_{01}\\g_{10}&g_{11}\end{pmatrix}.
$$

If $G$ acts on qubit $q$, then every assignment of the other $N-1$ bits defines one
independent update,

$$
\begin{pmatrix}\psi'_{a_0}\\\psi'_{a_1}\end{pmatrix}
=
G
\begin{pmatrix}\psi_{a_0}\\\psi_{a_1}\end{pmatrix},
$$

where $a_0$ has a zero at bit $q$ and $a_1$ has a one there. All other bits agree.
There are $2^{N-1}$ such assignments, hence $2^{N-1}$ disjoint $2\times2$ updates.

This already rules out the most literal implementation. Constructing
$I\otimes\cdots\otimes G\otimes\cdots\otimes I$ creates a $2^N\times2^N$ matrix to
perform work that is naturally expressed as small pairs. Reshaping the state into a
tensor is much better, but a straightforward reshape/permute implementation can still
allocate an output vector and copy data for every gate.

The optimized problem is simpler:

> Given a compact integer containing the $N-1$ non-target bits, reconstruct the two
> full $N$-bit indices of its amplitude pair.

## The amplitude-pair invariant

Throughout this chapter, basis-state indices and bit positions are zero-based. Bit zero
is the rightmost, least-significant bit. Julia arrays are one-based, so a Julia
implementation adds one only when indexing the array; that offset is not part of the bit
algebra.

For each compact index $i\in[0,2^{N-1})$, construct an anchor $a_0$ whose target bit is
known to be zero. The partner is then

$$
a_1=a_0\;\vert\;(1\ll q).
$$

Here $\ll$ is a left shift and $\vert$ is bitwise OR. OR states the intent directly:
set a bit that the construction guarantees is zero. The Julia notebook uses XOR to flip
the bit. Under the zero-bit invariant the two operations are equivalent, but OR makes
the precondition visible to the reader.

The pair has three useful properties:

1. $a_0$ and $a_1$ differ only at bit $q$.
2. Two different compact indices cannot generate the same anchor.
3. Every full index appears exactly once—as either an anchor or its partner.

Those properties are simultaneously the correctness proof and the reason an in-place
update is possible.

## Insert a zero bit

The compact index has every non-target bit packed together. To recover $a_0$, split it at
position $q$:

$$
\begin{aligned}
\text{low}  &= i\;\&\;\bigl((1\ll q)-1\bigr),\\
\text{high} &= (i\gg q)\ll(q+1),\\
a_0 &= \text{high}\;\vert\;\text{low}.
\end{aligned}
$$

The low mask preserves the $q$ bits below the insertion point. The right shift removes
those bits from the high part; shifting left by $q+1$ restores their positions while
leaving one new zero at $q$.

{% include figure.liquid path="assets/img/quantum-simulation/bit-insertion.png" alt="Worked binary example in which compact index 1011 is split into high and low bits, a zero is inserted at target position two to form 10011, and that bit is set to form partner 10111" caption="Zero-bit insertion for $i=1011_2$ and $q=2$. The compact index becomes $a_0=10011_2$; setting the inserted bit gives $a_1=10111_2$. The colored high and low groups retain their internal order." %}

For the worked example, $i=1011_2$. The two low bits are $11_2$ and the high bits are
$10_2$. Moving the high group left by one slot creates

$$
10\,0\,11_2=10011_2.
$$

Setting the inserted bit produces $10111_2$. Decimal labels—19 and 23—are less
illuminating than the binary view: the pair relationship is a single visible bit.

The Rust helper
[`insert_zero_bit`](https://github.com/nez0b/rustatevec/blob/main/crates/qsv-core/src/state/layout.rs)
uses an algebraically equivalent decomposition: clear the low part by shifting right and
back, recover the remainder, shift the high part once, and OR the pieces together. Its
unit tests check the inserted-zero invariant and verify that the generated pairs tile the
state space.

## Visit every pair exactly once

The whole gate kernel can be summarized without exposing language-specific machinery:

```text
for i in 0 .. 2^(N-1)-1
    a0 = insert_zero_bit(i, q)
    a1 = a0 | (1 << q)
    x0, x1 = ψ[a0], ψ[a1]
    ψ[a0], ψ[a1] = G · (x0, x1)
end
```

The order of the loads matters. Both old amplitudes must be read before either result is
stored. After that, writing in place is safe because no later iteration can refer to the
same pair.

A useful way to see the coverage proof is to run it backward. Take any full index $x$.
Clear bit $q$ to obtain its pair anchor. Delete that bit and the remaining $N-1$ bits form
one unique compact index $i$. Therefore every amplitude belongs to one generated pair,
and no amplitude can belong to two.

This optimization does **not** change the asymptotic cost of a generic one-qubit gate.
The simulator still reads and writes $2^N$ amplitudes, so the work remains
$O(2^N)$. The improvement comes from doing the necessary work directly:

- iterate over $2^{N-1}$ pairs rather than inspect or filter $2^N$ indices;
- avoid a $2^N\times2^N$ Kronecker operator;
- avoid per-gate permutation and output-vector allocation;
- preserve a simple, predictable access pattern;
- load each old pair once and store each new pair once.

## What the benchmark measures

To isolate that progression, I benchmarked one Hadamard gate in three Rust backends:

- **Independent oracle:** a correctness-first implementation with a separate output and
  gather/scatter logic that does not reuse the optimized index traversal.
- **Reshape + allocate:** a tensor-style intermediate that organizes the gate by blocks
  but allocates a full output state for each application.
- **In-place bit shift:** the direct $2^{N-1}$-pair traversal developed above.

{% include figure.liquid path="assets/img/quantum-simulation/rust-backend-comparison.png" alt="Two-panel point-range chart comparing one-gate runtime for independent oracle, reshape-and-allocate, and in-place bit-shift Rust backends at 12 and 16 qubits" caption="One Hadamard gate on an Apple M3 Pro with one Rayon thread and native CPU flags. Points are Criterion arithmetic means; horizontal intervals are 95% confidence intervals. The axis is logarithmic. This comparison intentionally excludes later CPU, threading, SIMD, diagonal-gate, and fusion optimizations." %}

| Qubits |      Independent oracle |      Reshape + allocate |       In-place bit shift |
| -----: | ----------------------: | ----------------------: | -----------------------: |
|     12 |    9.25 µs [9.21, 9.30] |    5.53 µs [5.50, 5.56] | **4.04 µs [4.01, 4.07]** |
|     16 | 161.3 µs [159.1, 164.2] | 110.8 µs [109.3, 112.9] | **64.7 µs [64.4, 65.0]** |

At 12 qubits, direct pair traversal is 1.37× faster than reshape-and-allocate and 2.29×
faster than the oracle. At 16 qubits those factors are 1.71× and 2.49×. These are not
universal constants: they describe these implementations, this gate, this compiler, and
this machine. The durable
[benchmark data](/assets/data/quantum-simulation/statevector-bitshift-benchmarks.json)
records the commit, command, toolchain, hardware, mean estimates, and confidence
intervals.

The independent oracle remains important even when it is slower. Differential tests run
random circuits and structured workloads through optimized backends and compare their
states with the implementation that does not share the same indexing trick. A fast
kernel and an independent correctness path serve different purposes.

## The bandwidth wall

Once pair enumeration is efficient, statevector simulation exposes its next limit: every
generic gate must move a large amount of amplitude data. More instruction-level
optimization cannot remove those reads and writes.

I encountered the same effect while working on
[QuEST issue #717](https://github.com/QuEST-Kit/QuEST/issues/717), opened by Tyson Ray
Jones. The issue targeted hot bit gather/scatter helpers. My merged
[PR #796](https://github.com/QuEST-Kit/QuEST/pull/796) added optional x86 BMI2 paths:
PDEP deposits packed bits into selected positions, while PEXT extracts selected bits in
mask order.

The important optimization was not merely replacing a short loop with an instruction.
Position masks and sortedness checks are invariant across the exponentially large
amplitude loop. Rebuilding or rechecking them for every amplitude can erase the benefit
or even regress performance. The useful design is therefore:

1. prepare masks and validate ordering once per gate or kernel;
2. use PDEP/PEXT inside the hot loop when the build and CPU support BMI2;
3. retain the scalar implementation as the portable fallback.

{% include figure.liquid path="assets/img/quantum-simulation/quest-optimization-dilution.png" alt="Horizontal interval chart showing QuEST BMI2 speedup shrinking from six-to-twelve times for isolated PDEP and PEXT helpers, to 2.5-to-3.9 times inside a gate kernel, to about one-to-1.3 times for whole circuits" caption="Three measurement levels from the work behind QuEST PR #796 on an Intel Xeon Gold 6448H. They are related but not interchangeable benchmarks: each broader level includes more statevector traffic and more work that BMI2 does not accelerate." %}

The isolated gather/scatter operation improved by roughly 6–12× in cache. Inside a real,
one-thread gate kernel, the gain was 2.53–3.89× because index generation now guarded
strided complex loads and stores. Across the reported 12- and 16-qubit circuits, the
measured range was 1.01–1.28×. With larger states or enough threads to saturate memory
bandwidth, the gain approached one and could occasionally become a regression.

That ladder is an example of Amdahl's law with a memory system attached. Accelerating
index arithmetic changes only the fraction of runtime spent generating indices. As the
state grows, irreducible amplitude traffic occupies a larger fraction of the gate. As
threads are added, they compete for finite bandwidth, so saved arithmetic no longer
translates proportionally into saved wall time.

PR #796 was merged into QuEST's `devel` branch on June 28, 2026 after maintainer review,
integration work, and upstream CI. The optional path is architecture-restricted; the
scalar fallback remains necessary both for portability and for processors where PDEP or
PEXT is not a fast hardware instruction.

## Where to optimize next

Zero-bit insertion is the foundation, not the endpoint. Once a simulator can enumerate
pairs correctly and without allocation, later layers can specialize the remaining work:

- traverse cache-friendly blocks rather than rely only on a flat index loop;
- recognize diagonal gates, which do not need a two-amplitude mix;
- add threading until memory bandwidth, not core count, becomes the limit;
- test SIMD against the actual layout rather than assume wider instructions must win;
- fuse adjacent gates when saved memory passes outweigh the extra local arithmetic.

Those optimizations deserve separate measurements because they answer different
questions. The central lesson here is smaller and more reusable: derive the data pairing
from the basis-index bits, make its invariant explicit, and arrange the loop so the
machine performs exactly one correct update per pair.

[Return to the Quantum Simulation Techniques series index →](/projects/quantum-simulation/)
