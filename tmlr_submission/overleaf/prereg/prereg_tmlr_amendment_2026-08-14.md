# Amendment 1 to the TMLR revision pre-registration

Written 2026-08-14, after `acebd1a` and **before any cell of the campaign is
generated or dispatched**. Amends R1 only. Nothing else in the parent document
changes.

## What prompted it

The parent registration says the RegMean sweep uses a grid "identical to E2",
which begins at lambda = 0. Before generating the cells we checked whether
lambda = 0 is defined for RegMean, and it is not, in a way that is a property
of the method rather than of our cohorts.

## The measurement

RegMean's data-free form solves, per layer,

    (sum_t w_t A_t^T A_t + lambda I) X^T = (sum_t w_t A_t^T A_t Delta_t)^T

in the **full input dimension**. Each `A_t` is `(r, in_dim)`, so each Gram has
rank at most `r` and their sum has rank at most `T*r = 64`, against `in_dim` of
4096 on the bases we use.

Probe at `in_dim = 512`, `T = 4`, `r = 16`, float64
(`scratchpad/regmean_lambda0.py`, reproduced in the campaign notes):

    rank(sum A^T A) = 64 of 512, exactly T*r
    eigenvalue 64 = 5.3e-03, eigenvalue 65 = 8.1e-18

    lambda      cond(M)     ||X||        note
    0           2.09e+19    53.73        numerically singular
    1e-8        2.35e+06     0.4331
    1e-6        2.35e+04     0.4331
    1e-3        2.45e+01     0.3958
    0.13        1.18e+00     0.0384

    reference: ||mean delta|| = 0.1029

At lambda = 0 the solve does not raise, which is the trap: LAPACK returns a
finite answer with a condition number of 1e19, and `||X||` of 53.7 against a
mean task delta of 0.103 is 500x amplification produced entirely by
floating-point noise in a 448-dimensional null space. It is not a large
measurement of a real quantity. It is not a measurement. In production the
deltas are half precision and the solve is float32, so the value would also
vary with hardware and thread count.

This is a genuine structural difference between the two solvers rather than an
inconvenience. Our encoder solves **inside** `range(Hbar)`, so its lambda = 0
cell is defined and its blow-up is a real conditioning effect. RegMean solves in
the full input space, so its lambda = 0 cell is not defined at all. The paper
will say this, because it bears directly on how far the family-level claim can
be pushed even if R1 confirms.

## The amendment

1. The RegMean grid is **1e-6, 0.01, 0.03, 0.05, 0.13, 0.30, 1.00**. Seven
   values, as registered; `1e-6` replaces exact `0`. Everything else about the
   design is unchanged.
2. `lambda = 1e-6` is named the **minimally regularised** reference, not the
   unregularised one, and is described that way in the paper.
3. **P2 is restated** in those terms: the excess at lambda = 1e-6 is worse on the
   shared arm than on the independent arm by a factor of at least 2, on at least
   3 of 4 bases. The threshold and the 3-of-4 requirement are unchanged.
4. P1 (ridge gain) is unchanged in form: `L(1e-6) - L(lambda*)`, larger on the
   shared arm on at least 3 of 4 bases.
5. The lambda = 0 cell is **not run**. Reporting a number produced by
   null-space noise, in either direction, would be worse than reporting none.
6. Because the two solvers' minimal-lambda references are not the same object,
   the comparison between our encoder's lambda = 0 column and RegMean's
   lambda = 1e-6 column is **not** a like-for-like comparison and will not be
   presented as one. Each method is compared against itself across arms, which
   is what both predictions are about.

## What this amendment cannot do

It was written before any RegMean cell existed, which is checkable in the commit
graph, and it changes a grid point rather than a threshold or a decision rule.
Had it moved a threshold, or had it been written after a single cell had landed,
it would be worth nothing and should be read as worth nothing.
