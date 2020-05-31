---
title: Notes About Network Propagation
author: Yiftach Kolb
date: \today
header-includes: |
    \usepackage{comment}
    \usepackage{amsmath}
    \usepackage{IEEEtrantools}
    \usepackage{listings}
abstract: Lecture Notes
---

## Algebra Background

A *Transition Matrix* is a real valued non negative square ($n^2$) matrix $A$ s.t. each of its
columns sums up to one: $$\forall j \sum_i A_{i,j} = 1,\ \forall i,j A_{i,j} \geq
1$$. That is when $A$ acts from the left:
$T(v) := Av, \forall v \in R^{n}$

A *state* is a non-negative vector $v \in R^n$ s.t $\sum_i v_i = 1$. If $v$ is a
state then $Av$ is also a state because: 
$$\sum_i(Av)_i = \sum_j v_j(\sum_k A_{j,k}) = \sum_j v_j \cdot 1 = 1$$

Because the columns of $A$ all sum to $1$, the columns of $A-I$ all sum to $0$.
Therefore $(1,1, \dots, 1)$ is an eigenvector of the rows with eigenvalue $1$.
Therefore there is also some columns eigenvector with eigenvalue $1$. 

Let $u \in R^n$ be such vector: $Au=u$. Let $u = u^+ + u^-$ such that $u_- =
\min(u_i,0)$ and the other one defined similarly for the non-negative components.

Because $A$ is non-negative $A(u^+) \geq 0, A(u^-) \leq 0$
And also $(A u)^+ = (u)^+$, so we must have $Au^+ \geq u^+$ 
(component wise). But if we had a strict inequality we would get:
$\|A(u^+/\|u^+\|)\| > 1$ which is a contradiction to $A$ being a transition
matrix.

Then $A u^+ = u^+, A u^- = u^-$ and one of them must be non-zero. It follows
that $A$ has a non-negative eigenvector with eigenvalue $1$ (one of $u^+, -u^-$
which is non-zero). If we normalized that eigenvector it becomes a state.

We can use similar reasonings to conclude that for any eigenvalue $\alpha$
of $A$ must it must hold that $|\alpha| \leq 1$.

If $A$ is strictly positive (entries are all positive), then it can only have
one eigenvector with eigenvalue $1$: Let $u,v$ s.t $Au=u, Av=v$. We choose $u,v$
to be states as we may do but what we have proven above.

Then let $w=u-v$. So $Aw = A(u-v) = u-v = w$. 
And $\sum_i w_i = 1 - 1 = 0$ by choice of $u,v$.

Like before $Aw^+ = w^+, Aw^- = w^-$
and because $w != 0$ but sums up to $0$, both $w^+, -w^- > 0$.
Because $w^-$ is non zero exactly on entries where $w^+$ is zero and vice versa, 
each of them must have at least one $0$ entry (and one none zero). But because
$A$ is strictly positive and $w^+$ is non-negative, $=Aw^+$ must have ONLY
positive entries, contradicting $Aw^+ = w^+$. It follows then that $u-v=0$ is
the only possible, hence the eigenvector with $1$ as eigenvalue is unique.

Finally we want to see that all the other eigenvalues are smaller in absolute
value than $1$. To that we can use Perron-Forbenious. Or we can see it directly:

Let $Aw = \alpha w$ with $|\alpha| = 1$. $A$ has a non-negative eigenvector with
eigenvalue $1$ and we can break $w = u + v$ such that $u \cdot v = 0$, $Au = u$.

Then $Au + Av = Aw = \alpha w = \alpha (u+v)$,
which we can rewrite into:

$$Av = \alpha u + \alpha v - Au = (\alpha - 1)u + \alpha v$$

Then because $u,v$ are orthogonal and $|\alpha|=1$ we get:

$$\|Av\|^2 = |\alpha - 1|^2\|u\|^2 + \|v\|^2 > \|v\|^2$$. Which contradicts $A$
being a transition matrix: For a transition matrix, for any $w \in C^n$ it must
hold that $\|Aw\| \leq \|w\|$. We only know that it's the case when $w \geq 0$
and $w \leq 0$.

Now let $w \in R^n$. We notice that element wise $(Av)^+ \leq Av^+$ and:

$$\|Aw\|^2 = \|(Aw)^+\|^2 + \|(Aw)^-\|^2 
\leq \|Aw^+\|^2 + \|Aw^-\|^2 \leq \|w^+\|^2 + \|w^-\|^2 = \|w\|^2$$

Now let $w \in C^n$. Then we can write $w = u + iv$ where $u,v \in R^n$.
Then $\|w\|^2 = \|u\|^2 + \|v\|^2$ and 
$$\|Aw\|^2 = \|Au\|^2 + \|Av\|^2 \leq \|u\|^2 + \|v\|^2 = \|w\|^2$$.


So to sums things up: if $T$ is a transition map, it has at least one
eigenvector with eigenvalue one which is unique if it is strictly positive. All
other eigenvalues must be absolutely smaller than $1$. $T$ maps states into
states.

Now let $T$ be strictly positive, so it's matrix has only positive entries.
$T$ has a unique eigenvector $u$ which is a state and $Tu=u$. All other
eigenvalues are in absolute term strictly less than $1$. Let $\gamma$ be the
second largest eigenvalue of $T$.

Let $s$ be a state vector. We can decompose: $s = \alpha u + \beta v$ sucht that
$u \perp v$. Then 
$$T^k s = T^k(\alpha u + \beta v) = \alpha u + \beta T^k v$$

Now $T^ks$ is a state because $T$ is transition and $s$ a state. But because $v$
is spanned by all the smaller eigenvectors it vanished: 
$$\|T^k v\| \leq |\gamma|^k \|v\| \to_{k \to \infty} 0$$.

It follows that $\alpha = 1$ otherwise $T^k s$ cannot be a transition. We have
shown namely that for any transition $s$, $T^ks$ converges to the stationary
state, and in particular since the columns of $A$ (the matrix of $T$) are
states, we have thar $A^k \to (u,u,u)$
