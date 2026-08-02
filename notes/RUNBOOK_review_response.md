# Dispatch runbook: answering the review

Everything below is written, validated off-cluster, and committed. Nothing has
run yet: the cluster (CLUSTER-HOST) is unreachable from the machine this was
prepared on, so `qsub` is the first step on each stage.

Order is deliberate. Stage 1 settles the review's main objection and is cheap.
Stage 2 probably deletes a weakness rather than conceding it. Stage 3 is the
one that can still move the whole story, so it starts last but has the longest
tail; start it as soon as a lane frees.

Prerequisite on the cluster, once:

```sh
cd ~/projects/rdmerge
git fetch && git checkout paper-consolidation      # or merge into phase3-bootstrap
python code/phase3/eval/downstream_metrics.py       # 22 cases
python code/phase3/merging/tests/test_rd_encoder.py # 10 cases
python code/phase3/merging/tests/test_knots.py      #  4 cases
python code/phase3/merging/tests/test_synthetic.py  # 17 property checks
```

---

## Stage 1 — W1, the merge-coefficient control (52 cells, ~4-5 h)

```sh
python code/phase3/scripts/gen_w1_alpha_sweep.py     # expect: 52 cells, validation OK
qsub code/phase3/scripts/pbs_w1_alpha_sweep.sh
# when done:
python code/phase3/scripts/analyze_w1_alpha.py
```

| group | cells | what it answers |
|---|---|---|
| `ta_alpha` | 28 | TA with weights `[alpha]*T`, alpha in {0.10 … 1.00}, 4 bases, seed1 |
| `rd_renorm` | 12 | rd-ridge at lambda\* with `renorm="ta"`: same direction, TA's norm |
| `rd_rank16` | 12 | rd-ridge at lambda\* at rank 16, storage parity with the baselines |

Sanity check before believing anything: the `alpha=0.25` column must reproduce
Table 1's TA row (Llama 0.213, Mistral 0.133, Qwen 0.104, Yi 0.099). If it does
not, the harness changed and everything downstream is suspect.

Thresholds are fixed in the analyzer. V1: W1 is upheld on a base if best-alpha
TA lands within 0.005 nats of rd-ridge; upheld on 2+ bases means the
contribution needs restating. V2: if renorm costs under 0.010 nats on 3+ bases,
the win is direction rather than scale. V3: rank-16 rd-ridge best on 4/4
answers the storage objection.

Offline prediction, from reconstructing W\*(lambda) exactly: at lambda\* the
method amplifies (alpha 1.02-1.31, norm ratio 1.3-2.4x) and 62-84% of its mass
is orthogonal to every rescaling of TA, and excess is *worst* where it most
resembles scaled TA (lambda=1.0 gives cos 0.99 and excess 0.289 vs 0.094 at
lambda\*). So V1 should come back "answered". If it comes back UPHELD, that
prediction was wrong and the paper's central claim is in real trouble.

## Stage 2 — W5 / A4 / A5, downstream re-score (144 cells, ~10-12 h)

```sh
python code/phase3/scripts/gen_w5_rescore.py         # expect: 144 cells
qsub code/phase3/scripts/pbs_w5_rescore.sh           # aborts if the fixes are absent
python code/phase3/scripts/analyze_w5_rescore.py
```

Re-runs every published downstream cell with both scorers fixed, into
`eval_downstream_v2/`. Published results are untouched, so the delta is
auditable. Cells now store full generations, so the next scorer change is a CPU
re-score rather than another 36 GPU-hours.

D1: if the (Llama-3.1, GSM8K EM) Spearman turns positive, the paper's one
"unexplained outlier" was an artifact, limitation (5) and the H1/H2/H3 appendix
come out, and W5 is answered rather than conceded. D2: if rd-ridge's across-seed
SD on its two unstable cells drops under 0.03 (published 0.074 and 0.084), the
"norm amplification yields degenerate greedy generations" explanation was wrong
too. D3: if the HumanEval spread narrows from 2-40x to under 10x, the "far more
deployable code merges" sentence must be rewritten.

## Stage 3 — A1, independent LoRA init (16 train + 28 eval, ~7 h + ~3 h)

```sh
python code/phase3/scripts/gen_a1_indep_init.py
qsub code/phase3/scripts/pbs_a1_indep_train.sh                       # stage 3a
python code/phase3/scripts/measure_subspace_geometry.py --cohort indep1   # 3b, CPU
qsub code/phase3/scripts/pbs_a1_indep_matrix.sh                      # 3c, gated on 16/16
```

Stage 3b is the actual readout and takes seconds. Compare against the committed
baselines, `results/phase3/subspace_geometry_{seed1,v1}.json`:

| cohort | median principal cosine | \|dA\| | sigma_max | soft d_eff / 64 |
|---|---|---|---|---|
| seed1 (published) | 0.995 – 0.997 | 0.16 – 0.19 | 1.999 | 16.3 |
| v1 (published) | 0.995 – 0.997 | 0.16 – 0.20 | 1.999 | 16.3 |
| indep1 | **?** | expect ~1.41 | expect ~1.0 | expect ~63 |

Two outcomes, both publishable:

- **Geometry opens up** (soft d_eff -> ~64). The floor-zero claim becomes true
  and attributable, but H-bar becomes well conditioned, so the sliver pathology
  that motivates the ridge should weaken and rd-ridge's margin may shrink. That
  is the risk, and stage 3c measures it directly. Better found now than in
  review.
- **Geometry unchanged.** Subspace collapse is a property of LoRA fine-tuning
  rather than of the shared init, the paper gains a control it currently lacks,
  and Appendix B's stated mechanism still needs rewriting, since it attributes
  the saturation to per-task loss geometry rather than to initialisation.

Note stage 3 also pins `seeds.data = 20260518`, matching the eval cells, so this
cohort is the first one without the ~14.5% / ~10% alpaca and magicoder
train/eval overlap (audit B1).

---

## Stage 4 — A2, KnOTS with a working inner merge (12 cells, ~1 h)

```sh
python code/phase3/scripts/gen_a2_knots_ties.py
qsub code/phase3/scripts/pbs_a2_knots_ties.sh        # aborts if test_knots fails
python code/phase3/scripts/analyze_a2_knots_ties.py
```

As shipped, `inner_combination="linear"` makes KnOTS algebraically Task
Arithmetic, because `Delta_t V V^T = Delta_t`. Published |KnOTS − TA| is 0.00003
to 0.00031 nats across the four bases, i.e. float noise. The paper cites that
agreement as evidence *for* its theory in four places (intro, §6.2 finding 3,
related work, App. J). This runs KnOTS-TIES, the variant the KnOTS paper
headlines, on the T=4 matrix at matched seeds.

Both outcomes force a rewrite. K1: if KnOTS-TIES differs from TA by more than
0.005 nats on 3+ bases, the published agreement was an implementation artifact
and all four claims go. K2: if it still tracks TA, the conclusion survives but
its evidence must be re-cited to these cells rather than to a no-op.

Not covered here: the T-scaling pool also carries KnOTS cells (App. D, "tracks
TA exactly at T = 7"). Same treatment needed before that sentence can stand.

## Stage 5 — W3, finite-rate sweep (24 cells, ~2 h)

```sh
python code/phase3/scripts/gen_w3_rate_sweep.py
qsub code/phase3/scripts/pbs_w3_rate_sweep.sh
python code/phase3/scripts/analyze_w3_rate.py
```

rd-ridge at b in {1,2,3,4,8,16} at each base's lambda\*, `realize="rank_deff"`,
seed1; b=32 already exists. rank_deff deliberately: truncating to rank 16 adds a
rate-independent error floor that would flatten the very slope being measured.

The encoder quantizes eta, which is (out x d_eff), at b bits per entry, so
2^{-2R/n} reduces to **2^{-2b}**: excess should fall 4x per bit, slope −2 in
log2 against b, matching the synthetic −2.00 ± 0.01. The analyzer fits the
quantization contribution `excess(b) − excess(inf)`, not raw excess, since
excess(inf) is the merge error and does not vanish with rate. b=1 is reported
but excluded from the fit as clipping-dominated.

Slope in [−2.4, −1.6] on 3+ bases gives the paper its first real-data
confirmation of the achievability exponent. Otherwise the abstract's implication
that the rate machinery does empirical work has to be withdrawn rather than
qualified. For contrast the analyzer also prints the published lambda=0 sweep
from `eval_e1/`, non-monotone on 4 of 4 bases and 9-11 nats on Mistral, which
has never appeared in the paper as a rate curve.

## Not yet scheduled

Needed for the review but not in these five stages:

- **W4** three extra 7-task subsets per base, 54 cells. Every T=7 point today is
  a single merge cell with one seed, and the Yi inversion gap (0.0197) is the
  size of the T=4 subset-to-subset range on the same base.
- **W4** a fourth base with high R and low saturation, ~40 cells. R and
  saturation are perfectly collinear on the current bases.
- **W6** seed provenance labels on every table cell. No compute.
- **W7** verify the remaining 31 bib entries. Two of two spot-checks were
  fabricated. No compute.
- **W8** build the anonymized audit bundle with `git filter-repo`; the history
  currently carries two personal emails, a resolvable GitHub handle and a
  cluster hostname naming the institution. No compute.
