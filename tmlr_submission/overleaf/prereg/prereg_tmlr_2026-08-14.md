# Pre-registration: TMLR revision campaign

Written 2026-08-14, in response to a referee report on the arXiv-format draft
(HEAD at the time: the paper-completion commit of 2026-08-14). Committed
**before** any cell of this campaign is generated or dispatched. The commit
ordering is the evidence; if any result commit of this campaign is an ancestor
of this file's commit, the registration is void and must be reported as void.

The report's central objection, R1, is that the paper's only surviving positive
claim is stated about a family of methods and tested on exactly one member of
that family, which we built. That is correct. Everything below is designed
around answering it, and around not being able to answer it in a way that
flatters us by accident.

Conventions carried over unchanged from the earlier registrations, so that
nothing here is a fresh choice made with the data in view:

- **Tie threshold** 0.005 nats. Fixed in the 2026-08-03 parent document.
- **Noise gate** one-directional: with SE = sd/sqrt(n) over cohorts, a win or a
  loss is downgraded to a tie unless |mean difference| > 2 x SE. The gate may
  only downgrade a verdict, never promote one.
- **Rank** pinned to `realize: rank_r` (rank 16) in every arm of every
  comparison unless the experiment is specifically about truncation (R7), so
  that audit finding A3 cannot reappear.
- Verdicts are reported in the branch the rule lands in, including
  `UNRESOLVED` and `PARTIAL`. We do not round to the nearer branch.

---

## R1 (PRIMARY, go/no-go). Does cohort conditioning affect a solver we did not build?

### Why this is the decisive experiment

Section 7.2 currently reports that methods which solve a linear system are
severely affected by cohort conditioning, and Section 2 names RegMean, LoRM and
RegMean++ as that family. The only solver actually swept is our own encoder.
The referee is right that this is an attribution to a family from a single
member, and that the member is ours.

RegMean (Jin et al., 2023), in the data-free adapter-only form already
implemented in `code/phase3/merging/regmean.py`, is a solver we did not design,
with a ridge term that is part of its own published formulation rather than
something we added. It is therefore the right test.

### Design

The E2 design, with the method swapped and nothing else touched:

- 4 bases x 7 lambdas x 2 cohorts = **56 cells**.
- Lambdas: 0, 0.01, 0.03, 0.05, 0.13, 0.30, 1.00. Identical grid to E2.
- Cohorts: `seed1` (shared) and `indep1` (independent). Identical to E2.
- Both arms built from one template with only the adapter directory swapped, so
  tasks, n_eval, max_seq_length, evaluation seed and VRAM gate are byte
  identical across arms.
- `realize: rank_r` in every cell.

### Predictions, fixed now

Both are stated as the E2 predictions were, and both are about the *shape* of
the response rather than its size, because we have no basis for predicting the
size on a method we have not swept.

- **P1 (ridge gain).** `L(0) - L(lambda*)` is larger on the shared arm than on
  the independent arm, on at least 3 of 4 bases.
- **P2 (unregularised penalty).** The lambda = 0 excess is worse on the shared
  arm than on the independent arm by a factor of at least 2, on at least 3 of 4
  bases.

### Decision rule

- **CONFIRMED** if P1 and P2 both hold on at least 3 of 4 bases.
- **REFUTED** if neither holds on at least 3 of 4 bases.
- **PARTIAL** otherwise. We then claim only the half that held, in the same
  words E2 used for its own partial outcome.

### What we write in each branch, fixed now

This is the part the referee asked for explicitly, and it is the part that is
worth nothing if written afterwards.

- **CONFIRMED.** The family-level claim stands and is stated as tested on two
  solvers, one of which we did not design. Section 2's sentence about the
  closed-form family is kept, with the evidence named.
- **PARTIAL.** The claim is narrowed to the half that replicated, in the
  abstract, Section 1, Section 7.2 and Section 8, and Section 2's family
  sentence is cut to the property that actually replicated.
- **REFUTED.** The conditioning claim is narrowed to our own encoder
  **everywhere**: abstract, Section 1, Section 7.2, Section 8, and the Section 2
  sentence naming RegMean, LoRM and RegMean++ as the sensitive family. We state
  in the paper that the effect did not replicate on the one solver we did not
  build, and that the paper's operational content is correspondingly reduced. We
  do not report the RegMean arm as a footnote or an appendix curiosity.

We record now, before seeing any cell, that REFUTED is the outcome that
requires the largest rewrite and is also the outcome the reader most needs.

### Secondary, not part of the verdict

RegMean++ and LoRM are not run. RegMean++'s distinguishing mechanism is a
cross-layer dependency in the activation Gram; LoRM's setting is federated
continual learning. Neither has a data-free adapter-only form that we could
implement without inventing the parts that make it that method, and inventing
them and then reporting the result under their name would be worse than not
running them. The paper will say this rather than implying the family was
covered.

---

## R4. Is the repaired KnOTS a real arm?

`eval_a2_knots_ties` already holds 12 cells (4 bases x seed1/2/3) of KnOTS with
`inner_combination="ties"`, run before this registration and not read against
any rule. This registration governs how they are used, and the 12 independent
cohort cells that complete the design.

- **Design.** Add 4 bases x indep1/2/3 = **12 cells**, so the repaired KnOTS
  exists in both arms at n = 3.
- **Rule.** The repaired KnOTS enters the null of R3 as an independent arm if
  its worst-task excess differs from task arithmetic by more than the tie
  threshold on at least 3 of 4 bases, on the shared arm. Otherwise it is
  removed, and the null is restated over the 16 cells that remain.
- Either way, the **default-configuration KnOTS is removed from the null**. The
  referee is right that a method which is task arithmetic to four decimals is
  the same data point counted twice, and our Appendix E argument for keeping it
  is withdrawn.

---

## R3. The null, with a gate and a minimum detectable effect

No new cells. `eval_matrix_seeds` (shared, seed1/2/3) and `eval_a1_indep`
(independent, indep1/2/3) already hold three cohorts per arm, so the gate that
could not be computed at n = 1 can be computed now.

- **Design.** Per (base, method), the difference is the mean over three cohorts
  of independent minus shared, with SE = sd/sqrt(3).
- **Rule.** Apply the standard one-directional gate. A cell counts as an effect
  only if |mean difference| > max(0.005, 2 x SE).
- **Reported regardless of outcome:** the minimum detectable effect, defined as
  the smallest |mean difference| that would have cleared the gate in each cell,
  and the conclusion phrased as "no effect detectable at X nats" with X named,
  never as "no effect".
- The current text's mean difference of 0.0014 nats sits above the paper's own
  stated metric resolution of about 0.001 nats. Whatever the gate returns, that
  tension is resolved explicitly in the text rather than left standing.

---

## R7. Does the conditioning link survive without rank truncation?

Section 6.4 states that theory, geometry and experiment constrain three
different objects, and that we expect the story to survive truncation but have
not shown it. This closes that.

- **Design.** The E2 ridge sweep repeated with `realize: rank_deff`
  (untruncated), 4 bases x 7 lambdas x 2 cohorts = **56 cells**. Identical in
  every other respect to E2, so the truncated sweep is the comparison arm and is
  not re-run.
- **Rule.** The link **survives** if, untruncated, the lambda = 0 shared-versus-
  independent ratio is at least 2 on at least 3 of 4 bases, and the ridge gain
  remains larger on the shared arm on at least 3 of 4 bases. It **fails** if
  either falls below on at least 2 of 4 bases.
- **If it fails**, every sentence linking the measured kappa to the observed
  degradation is downgraded to a conjecture, in the abstract, Section 1,
  Section 6.4 and Section 7.2, and the paper says the link was tested and did
  not hold.

---

## R8. The ridge sweep gets its gate

- **Design.** The E2 sweep extended to `indep2` and `indep3`: 4 bases x 7
  lambdas x 2 cohorts = **56 cells**, giving n = 3 on the independent arm.
- **Rule.** With three independent cohorts, lambda* is selected per cohort and
  the ridge gain is averaged across them, with the standard gate applied to the
  shared-versus-independent gain difference. A gain difference that does not
  clear the gate is reported as a tie even where the raw margin is large.
- The shared arm remains n = 1 (seed1). We state that, rather than implying the
  gate is two-sided.

---

## R6. Merge results at T = 3

- **Design.** The independent-cohort merge matrix re-run with the degenerate
  translation adapter dropped: 4 bases x 7 methods x indep1/2/3 at T = 3,
  **84 cells**, minus any method removed by R4.
- **Rule.** The primary result remains the T = 4 matrix, as run and as
  registered in the 2026-08-03 document. T = 3 is a robustness arm. If any Q1
  per-base verdict (win, tie, loss) differs between T = 4 and T = 3, both are
  reported in Table 3 and the difference is stated in the text; the T = 4
  verdict is not replaced.
- **If the budget does not permit these cells**, the alternative is fixed now
  and is not a judgement call made later: Section 7 and every affected table
  caption state that all reported worst-task NLL excess figures include an
  adapter that never learned its task, and that no T = 3 merge results exist.

---

## S3. A margin-aware stability control, registered before the re-run

Section 7.3 withdrew the benchmark-confounding claim on a control that counts a
top-1 change as instability regardless of margin. Both flips involve methods
separated by far less than the tie threshold (0.0140 against 0.0141 on Qwen).
We declined to rewrite that rule after seeing the outcome, and we still
decline. This registers a second control instead.

- **Design.** Same three independent cohorts, same top-1 comparison, no new
  cells.
- **Rule.** A base counts as unstable only if its top-1 method changes across
  cohorts **and** the margin between the top two methods exceeds the tie
  threshold of 0.005 nats in at least one cohort where the change occurs.
- **Verdict.** If at most 1 of 4 bases is unstable under this rule, the
  benchmark-confounding claim is reinstated as **provisional**, with both the
  original margin-blind verdict and this one reported side by side, and with the
  original identified as the one registered first.
- Under no outcome is the original verdict deleted or replaced. The paper
  reports that the first control withdrew the claim and that a second,
  margin-aware control, registered afterwards and named as such, returned
  whatever it returns.

---

## Protocol for the whole campaign, fixed now

1. Every generator asserts that both arms differ only in the intended variable,
   and refuses to write cells otherwise.
2. A one-cell smoke test runs before each wave. The smoke inspection checks
   configuration equality and that the method is not a no-op; it does not look
   at the direction of any effect. Where a smoke cell is also a verdict cell,
   we say so and report blindness as n-1 of n.
3. Analyzers are committed **before** the cells they read have landed, as in
   the 2026-08-03 and 2026-08-07 campaigns.
4. If any verdict in this campaign contradicts a published verdict of ours, both
   are reported, with the superseded one named and dated, in the paper rather
   than only in this repository.
5. Nothing in this campaign re-runs the TIES, DARE, TVQ or task arithmetic
   reference cells. They are the pre-existing ones.
