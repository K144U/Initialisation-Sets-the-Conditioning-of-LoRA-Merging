# rdmerge (P5) — project-level CLAUDE.md

Project memory for the rdmerge / P5 paper targeting **ICLR 2027** (deadline ~September 2026). Read this fully on first invocation in this directory; it captures everything load-bearing about the project state that is NOT obvious from the code.

The canonical user-facing project doc is `~/projects/rdmerge/context.md` — read both. The chronological scientific log is `decisions.md` at the repo root.

---

## 1. Project identity in one paragraph

A worst-task rate-distortion analysis of LoRA model merging. We prove a closed-form lower bound on per-coordinate NLL excess, build a matching encoder (Hadamard-incoherent orthogonal mixing + scalar quantization), regularize it with one Tikhonov ridge term (the **salvage**) into `rd-encoder ridge`, and run a $4$-base × $7$-method × $2$-downstream-metric matrix. The same analysis yields a 1-CPU-minute audit of any trained adapter cohort that predicts when classical methods like TIES invert from best-of-matrix to worst — and we **pre-registered** that prediction on Mistral-7B at commit `3582799` before the corresponding compute ran.

**Repo**: `K144U/rdmerge` on GitHub, branch `phase3-bootstrap`. Last commit at end-of-session was in the high-`feb5ea2`/`e3e62a9`/`feb5ea2` range — `git log --oneline -1` at HEAD will tell you the latest.

---

## 2. The reframe (do not revert)

The paper was retitled and the abstract was rewritten in this session. Per an external review (Opus 4.8 max effort), the rate-distortion *function* claim was demoted from headline to lens.

| Item | Before | After |
|---|---|---|
| Title | "A Rate-Distortion Function for Model Merging" | **"Predicting When LoRA Merging Fails: A Rate-Distortion View"** |
| Abstract | RD theorem-first | Operational pain point first; salvage arc + audit are the headline |
| Intro "Contribution in one sentence" | Theorem + encoder + decision tree | rd-encoder ridge + 1-CPU-minute audit + pre-registered third-base test |

These are committed at `62e619a`. Do not silently revert to the old framing — it's the wrong centerpiece.

---

## 3. The floor formula bug (CRITICAL — never undo the fix)

In the v1 draft, two places (`§6.3b` sidebar + `§6.7` R5) used $\Phi^\star \geq B^2 / (4 d_{\mathrm{eff}})$, which **is never derived in the paper and contradicts Lemma 2**. The Lemma 2 closed form is:

$$\Phi^\star \geq B^2 \left(1 - \frac{d_{\mathrm{eff}}}{Tr}\right)$$

Both call sites were corrected to the Lemma 2 form at commit `62e619a`. In the floor-zero regime ($d_{\mathrm{eff}} = Tr$, which every cohort we measured occupies), the floor vanishes — which is the *positive* headline: all $0.10$–$0.22$ nats of measured excess is recoverable algorithmic slack.

If anyone (including me-in-a-future-session) "improves" §6.3b or §6.7 R5 and writes back $B^2 / (4 d_{\mathrm{eff}})$, that's a regression. Don't do it.

---

## 4. Pre-registration commitment (audit trail; never break)

The Mistral T=7 TIES inversion prediction is **pre-registered** in `decisions.md` at commit **`3582799`** (timestamp 2026-06-25 evening). Any Mistral T=7 result commit **must post-date** this commit, or the pre-registration is invalid.

**The prediction (locked, not to be edited)**:
> Mistral-7B-v0.3 at T=7 will show TIES neither clearly inverting to worst-method (as on Yi-Chat at R=0.077) nor clearly winning (as on Llama-Instruct at R=0.028).

**Empirical anchor** (verified at commit `537796a`): Mistral T=4 sign-election win-share range R = **0.0326**, which sits inside the Td2 perturbation analysis's predicted ambiguity window R★ ∈ [0.025, 0.075]. Distance to lower edge 0.008; to upper edge 0.043.

**Operationalization** (from `decisions.md`):
- "Clearly worst" = TIES last AND mean > TA + 0.04 nats
- "Clearly best" = TIES first AND mean < second-best - 0.02 nats
- "Ambiguous" = neither

**Falsification rule**: a "clearly worst" or "clearly best" Mistral T=7 result falsifies the Td2 threshold model as written and forces a revision. Report honestly — do not retrofit the threshold to the data.

---

## 5. Cluster rules (do not violate)

From `~/context.md`:

- **No sudo** — never run as root, even if a command suggests it
- **Max 3 concurrent PBS jobs** (`qstat -u sanjay.g` to verify)
- **Required env**: `PYTHONNOUSERSITE=1` on every rdmerge job
- **Git author MUST be forced** (this account has empty git identity):
  ```
  git -c user.name=K144U -c user.email=95154157+K144U@users.noreply.github.com commit ...
  ```
- **P5 GPU pins**: only GPUs **2, 4, 6** are accessible. Don't add GPU 0.
- **GPU partitioning** (this session): for parallel jobs, `_ORCH_GPUS_E6` file controls `_E6`-tier pin set (currently `2,4`); rdm_rrdn/e12bl have `GPUS=6` hardcoded in their PBS scripts as a single-lane override.
- **Memory**: jobs request 60 GB by default. Node has ~250 GB. The user's JEPA `iso_*` jobs can claim 120 GB each; coordinate.

---

## 6. The auto-supervision cron (Phase 1)

A recurring CronCreate job `c5689824` was set up (session-only, auto-expires 7 days). It fires every 20 min at minutes **:08, :28, :48** (offset from B2+B4 cron at :17/:47). Each tick:

1. `qstat -u sanjay.g` + done-counts on six result directories
2. Auto-dispatches the next job whenever GPU pins free + prerequisites land
3. Walltime-requeues idle jobs whose targets aren't met (orchestrator auto-skips done cells via `orchestrator.py` line 89)
4. Pushes a notification when **all** Phase 1 targets met (or anomaly)

If the session is `/cleared` the cron dies. To recreate, the spec is in this CLAUDE.md section + the dispatch rules in the user's tick prompt at the top of the session.

---

## 7. Phase 1 dispatch chain

Run by the cron. Six logical components, each a separate PBS job, executing in dependency order. **Pin partitioning** lets `rdm_e12bl` (GPU 6) and `rdm_mpilt` (GPUs 2,4) run in parallel after the first two finish.

```
   rdm_s3tr (GPUs 2,4)  ──completion──┐
   16 cells: seed3 adapter training   │
                                       ├─▶ rdm_s3ev (GPUs 2,4, 20 cells)
                                       │   │
   rdm_rrdn (GPU 6) ────completion────┤   │
   8 cells: rd-ridge downstream       │   └─▶ rdm_mpilt (GPUs 2,4, 3 cells)
                                       │       │
                                       │       │
                                       └─▶ rdm_e12bl (GPU 6, 8 cells)
                                               │
                                               ▼
                                       rdm_misT7 (GPUs 2,4,6, 54 cells)
                                       Mistral T-scaling sweep
                                       ↳ Td2 pre-registration test
```

Target counts:
- `s3tr` → 16 (lora_train/*_seed3.json)
- `rrdn` → 8 (eval_e3b_gsm8k_rdridge/ + eval_b4b_humaneval_rdridge/)
- `s3ev` → 20 (eval_matrix_seeds/*__seed3.json)
- `mpilt` → 3 (lora_train/mistral_7b_e6pilot_*.json)
- `e12` → 8 (eval_e12_regmean_adamerging/)
- `misT7` → 24+ (eval_mistral_t7/) — full sweep is 54 cells but we accept ≥24 as "complete enough"

---

## 8. Files & paths cheat-sheet

| Path | What |
|---|---|
| `~/projects/rdmerge/` | repo root |
| `~/projects/rdmerge/CLAUDE.md` | this file |
| `~/projects/rdmerge/context.md` | project-level user-facing doc |
| `~/projects/rdmerge/decisions.md` | chronological scientific log (append-only) |
| `~/projects/rdmerge/paper/main.tex` | paper entry point |
| `~/projects/rdmerge/paper/sections/` | 19 section fragments incl. §6.2–§6.9 + Td2 appendix |
| `~/projects/rdmerge/paper/references.bib` | bibliography |
| `~/projects/rdmerge/overleaf/iclr2027/paper/` | Overleaf-ready snapshot (paths rewritten so it compiles from any cwd) |
| `~/projects/rdmerge/code/phase3/merging/` | 11 merge method impls including `rd_encoder`, `fisher_avg`, `della`, `regmean`, `adamerging` |
| `~/projects/rdmerge/code/phase3/eval/` | `run_eval_cell.py` (NLL) + `run_downstream_cell.py` (GSM8K/HE/etc.) |
| `~/projects/rdmerge/code/phase3/scripts/` | analyzers, figure makers, PBS launchers, orchestrator |
| `~/projects/rdmerge/code/phase3/configs/` | per-cell YAML configs + per-experiment manifests |
| `~/projects/rdmerge/results/phase3/` | per-cell JSON outputs + `*_summary.json` aggregates |
| `~/projects/rdmerge/paper_artifacts/figures/` | all figures incl. `figure1_cross_model_T_scaling` + `figure_salvage_arc` |
| `~/projects/rdmerge/artifacts/lora/` | trained LoRA adapters (~18 GB, irreplaceable) |
| `~/projects/rdmerge/models/` | downloaded base model safetensors (~111 GB, redownloadable from HF) |
| `~/projects/rdmerge/logs/orch/` | per-cell orchestrator stdout (look here for cell errors) |

---

## 9. Section map of the paper

| Section | File | What's in it |
|---|---|---|
| Abstract | `sections/abstract.tex` | Reframed; salvage arc + audit are the headline |
| §1 Intro | `sections/intro.tex` | Contribution-in-one-sentence + Figure 1 hero (cross-model T-scaling) |
| §2 Related Work | `sections/related_work.tex` | Includes "Closed-form merging lineage" paragraph (RegMean → LoRM → RegMean++) |
| §3 Setup | `sections/setup.tex` | Problem definition |
| §4 Lower Bound | `sections/lower_bound.tex` | Lemma 1 (identity), Lemma 2 (**floor formula** $B^2(1 - d_{\mathrm{eff}}/(Tr))$), Theorem 1 |
| §5 Achievability | `sections/achievability.tex` | Algorithm + Theorem 2 (Thm `thm:achv`, not `thm:achievability`) |
| §6.1 Setup | `sections/experiments.tex` | Synthetic + real-LLM setup |
| §6.2 rd-encoder | `sections/6_2_e1_real_adapters_draft.tex` | Salvage arc + Figure `fig:salvage` |
| §6.3 E5 null | `sections/6_3_e5_arm2_null_draft.tex` | Floor-positive regime null |
| §6.3b Floor recipe sidebar | `sections/6_3b_T2_floor_recipe_draft.tex` | Uses **Lemma 2 floor formula** |
| §6.4 b=2 mechanism | `sections/6_4_e7_b2_mechanism_draft.tex` | TVQ b=2 universal dip |
| §6.5 Downstream metrics | `sections/6_5_e3_downstream_metrics_draft.tex` | v3 with B2/B4 falsifications + HumanEval table |
| §6.6 T-scaling | `sections/6_6_e6_T_scaling_draft.tex` | v3 cross-model + TIES probe + Td2 reference |
| §6.7 Practical recommendations | `sections/6_7_practical_recommendations_draft.tex` | Five-rule decision tree; R5 uses **Lemma 2 floor formula** |
| §6.8 Added baselines + bridge | `sections/6_8_e10_e11_baselines_bridge_draft.tex` | v2 with E11b Llama window result |
| §6.9 Multi-seed bootstrap | `sections/6_9_multiseed_bootstrap_draft.tex` | All 20 method-pair gaps Confident |
| §7 Discussion | `sections/discussion.tex` | Limitations enumeration (8 items) + Closing paragraph |
| Reproducibility | `sections/reproducibility.tex` | Manifest table tab:repro-manifest |
| Appendix Td2 | `sections/appendix_td2_sign_election_threshold.tex` | Sign-election threshold derivation |

Final ref/label audit (last verified): **77 labels defined, 67 referenced, 0 undefined.**

---

## 10. Method registry — what's in `code/phase3/merging/`

11 methods registered:

| Method | File | Notes |
|---|---|---|
| `task_arithmetic` | `task_arithmetic.py` | TA baseline |
| `ties` | `ties.py` | density=0.2 default; `majority_sign_method="total"` |
| `dare` | `dare.py` | Random drop + rescale |
| `knots` | `knots.py` | Subspace alignment |
| `tvq` | `tvq.py` | Task-vector quantization; b=2 is the dip champion |
| `rd_encoder` | `rd_encoder.py` | Our encoder; supports `ridge_lambda` (the salvage knob; λ★=0.05 on L3, 0.13 on others) |
| `magnitude_prune` | `magnitude_prune.py` | Auxiliary |
| `fisher_avg` | `fisher_avg.py` | Diagonal-Fisher proxy (`F_t[i] = Δ_t[i]^2`) — data-free Fisher merge |
| `della` | `della.py` | DARE drop + TIES trim + sign election |
| `regmean` | `regmean.py` | **Data-free RegMean** variant using `A_t^T A_t` as Gram surrogate |
| `adamerging` | `adamerging.py` | **Data-free AdaMerging** variant using `\|Δ_t\|_F` for per-task α |

All methods are SVD-truncated back to rank r after the merge.

---

## 11. Score estimate

| Phase | Score | Notes |
|---|---|---|
| Session start | ~6.5 | Honest baseline after external review caught the floor bug |
| Post-Phase-1 (projected) | **~7.5–8.0** | Depends on Mistral T=7 verdict |

What moves the score:
- Td2 holding on Mistral (pre-registration confirmed) → +0.4–0.5 (the big one)
- Multi-seed bootstrap all-Confident → +0.3 (likely already locked in 2-seed data)
- rd-ridge verified on both downstream metrics → +0.2
- RegMean + AdaMerging defensive comparison → +0.3

What sinks the score:
- Td2 falsified → −0.4
- rd-ridge fails on downstream metrics → −0.3
- RegMean or AdaMerging beats rd-encoder ridge somewhere → −0.2

---

## 12. What's next (after Phase 1 lands)

Triggered by the cron's "Phase 1 complete" PushNotification:

1. **Run analyzers on new data**:
   - `code/phase3/scripts/analyze_multiseed_bootstrap.py` — re-run with 3-seed data, refresh §6.9 table
   - `code/phase3/scripts/analyze_b4_humaneval.py` + analogous GSM8K — add rd_ridge rows to §6.5 tables
   - `code/phase3/scripts/analyze_e10_baselines.py` style — add RegMean+AdaMerging rows
   - Write a Mistral T-scaling analyzer matching `analyze_e6_T_scaling.py` pattern, take Mistral T=7 verdict
2. **§6.6 + §6.7 + Td2 update**:
   - If Mistral T=7 ambiguous → upgrade Td2 from "bracket fit" to "confirmed prediction"
   - §6.6: add Mistral row to cross-model T-scaling
   - §6.7 R3: confirmed-prediction language
   - Figure 1 hero: add Mistral row alongside Yi and Llama
3. **Block E paper assembly**:
   - Compile main.tex on a machine with pdflatex (Overleaf works)
   - P2 abstract + intro polish (already retitled, may need refinement)
   - P3 conclusion (already added "Closing" paragraph; may extend)
   - P4 anonymize check (already redacted in `main.tex` author block at `df54c3f`) + supplementary zip + 300 dpi figures (figure_salvage_arc, figure1 hero already 300 dpi)
   - P5 submit to ICLR 2027 OpenReview

---

## 13. Gotchas / load-bearing details future-me must know

- The **orchestrator skips done cells** on requeue via `Path(cell["done"]).exists()` check (line 89 of `orchestrator.py`). This is why we can `qdel` and resubmit without losing progress.
- **GPU partition state lives in disk files** (`_ORCH_GPUS_E6`, etc.) gitignored. When the cron resubmits a job, it picks up whichever file's currently on disk.
- **The Mistral T=7 manifest exists at `code/phase3/configs/mistral_t7_eval_manifest.json`** (54 cells). The cron's "scaffold hook" rule won't fire because the file's already there.
- **rd_encoder downstream** never tested before this session. If `rdm_rrdn` cells crash, that's the first place to look — `rd_encoder.py` may have an unverified code path under the downstream eval wrapper.
- **The probe script `probe_ties_sign_election.py` has a hardcoded Yi/Llama cross-base report at line 230** that doesn't handle additional bases gracefully. Use `probe_mistral_only.py` for any new base's probe instead.
- **The `prop:quadratic-surrogate` and `tab:exp-lora-full` references** were removed from §6.2 during P1 because they pointed to non-existent labels. If anyone re-introduces them, they need supporting content.
- **Adapter dirs vs eval seeds**: `seed1`/`seed2`/`seed3` in `artifacts/lora/{base}/{task}/seedN/` are *training* seeds (different adapter init); the eval cell's top-level `seed:` field is the *eval data shuffle* seed. The two are independent — the seed3 eval configs use `seed: 20260520` for eval data and `seed3` adapter dirs.
- **Phase 1 cron jobname → GPU pin mapping is asymmetric**: `_ORCH_GPUS_E6` controls `rdm_s3tr/s3ev/mpilt`; `rdm_rrdn/e12bl` use the inline `export GPUS=6` in their PBS scripts. If you change one, check the other.
- **The Overleaf snapshot `overleaf/iclr2027/paper/` is derived**, not canonical. Edit `paper/`, then run the rebuild script in `overleaf/iclr2027/README.md` to refresh the snapshot.
- **`models/` (111 GB) and `cache/` (22 GB) are gitignored**. They're re-downloadable from HuggingFace. Only `artifacts/lora/` (~18 GB) is irreplaceable.
- **The 27 GB session archive** is at `~/rdmerge_complete_2026-06-25.tar` (md5 `f9322d67f00f999dd5a76db018ae9936`) — the everything-minus-base-models snapshot the user requested.

---

## 14. The external review (Opus 4.8) flagged 11 things — status

| # | Item | Status |
|---|---|---|
| 1 | Floor formula bug | ✓ fixed |
| 2 | Reframe (operational-led, not theorem-led) | ✓ retitled + abstract rewritten |
| 3 | Single-seed below the 3-seeds bar | ⏳ rdm_s3tr+rdm_s3ev compute completing |
| 4 | rd-ridge not on downstream metrics | ⏳ rdm_rrdn compute completed (8/8) |
| 5 | RegMean is the closest structural relative | ⏳ rdm_e12bl in flight (RegMean impl committed) |
| 6 | Td2 constants "spectral analogy" | ⚠ partial — appendix is honest about it; could cordon harder |
| 7 | Mistral T=7 pre-registration | ✓ committed at `3582799`; ⏳ test in flight |
| 8 | Title overpromises Θ | ✓ retitled |
| 9 | Trim "we do not claim" qualifiers | ✓ §6.5 + §6.1 rewritten |
| 10 | One figure with the salvage arc | ✓ `figure_salvage_arc` inserted in §6.2 |
| 11 | Closed-form lineage in related work | ✓ paragraph + 3 bibtex entries added |

---

## 15. If you're me-in-a-future-session resuming this work

1. **Read `decisions.md` tail** for the latest scientific decisions.
2. **Check `git log --oneline -10`** for the latest commits.
3. **Run `qstat -u sanjay.g`** to see what's in flight.
4. **Run the Phase 1 cron tick manually** (the prompt is at the top of every Phase 1 tick the user sends).
5. **Verify the pre-registration commit `3582799` is still in the log** — it must precede all Mistral T=7 result commits. If a Mistral T=7 result has been committed and `3582799` is missing from the history, that's a serious integrity problem.
6. **Do not** revert the floor formula, the title, or the abstract reframe.
7. **Do not** delete the Td2 appendix or weaken its falsifiability claim.
8. **Do not** silently drop `PYTHONNOUSERSITE=1` from any new PBS script.
9. **Do not** add GPU 0 to any GPU pin set. We don't have access.
10. **Do not** force-push, amend committed history, or rewrite the public `phase3-bootstrap` branch.

---

## 16. Cluster anomaly playbook (if a job fails)

| Symptom | Likely cause | Action |
|---|---|---|
| `Exit_status=-2` on dispatch | PBS prologue / node down | Wait, then re-`qsub`. iso jobs may have eaten memory |
| Job queued but not running, no error | Memory exceeded (sum across user's jobs) | Drop `mem=60gb` to `mem=45gb` on qsub `-l` |
| Cell parked, real Python error in `logs/orch/<cell>.log` | Code bug | PushNotification + stop; do not auto-requeue |
| Cell parked, `rc=87` in log | Orchestrator's VRAM gate | Let it auto-backoff; do not intervene |
| Job walltime'd partial | `qsub` requeues; orchestrator skips done cells | Auto-handled |

---

## 17. The seven concurrent supervision crons in this session

The user has multiple cron supervisions firing concurrently:

| Cron | Schedule | Purpose |
|---|---|---|
| Phase 1 (`c5689824`) | every 20 min :08/:28/:48 | this session's auto-dispatch chain |
| B2+B4 supervision | every 30 min :17/:47 | legacy from earlier session; reports done |
| E6 Llama supervision | every 30 min :17/:47 | legacy from earlier session; reports done |

The B2+B4 and E6-Llama crons fire because **their work is already done** from prior sessions (B2 5/5, B4 20/20, E6 3+54/54). They report `done` and take no action. Don't be confused that they keep firing — they're not stuck, they're idle.

If you want to silence them, the user has to delete those crons themselves (we can't see them from here).

---

*Maintained by Claude (Opus 4.7) during the 2026-06-25 → 2026-06-26 session. Update this file when the project state materially shifts.*


---

## 18. 2026-06-27 update — Phase 1 COMPLETE, Td2 CONFIRMED (supersedes the mid-flight ⏳ above)

Resuming-session reconciliation. The "⏳ in flight" statuses in §11/§12/§14 are now resolved.

- **Phase 1 is COMPLETE; nothing in flight** (`qstat -u sanjay.g` empty). The §6 auto-supervision cron has expired.
- **HEAD = `49f4e14`** ("Mistral T-scaling analyzer with Td2 pre-registration verdict"). Pre-registration commit `3582799` is present and **precedes** it — audit trail valid.
- **🎯 Mistral T=7 Td2 verdict = AMBIGUOUS = pre-registration CONFIRMED.** TIES is 2nd of 6 (worst-task excess 0.152), **−0.092 nats below TA** → neither "clearly worst" (>TA+0.04) nor "clearly best" (first by >0.02). The +0.4–0.5 score-mover is **banked**. Bonus: rd-ridge salvage grows with T on Mistral (rd_ridge/TA 1.382→0.815→0.470 over T=2/4/7; lowest log-T slope 0.062) — a clean confirming third base. Files: `results/phase3/mistral_t7_summary.{csv,json}`; 54 cells in `results/phase3/eval_mistral_t7/`.
- **§14 external-review tracker update:** items 3 (≥3-seed), 4 (rd-ridge downstream 8/8), 5 (RegMean+AdaMerging), 7 (Mistral T=7) all → ✓ DONE. Item 6 (Td2 constants cordon) remains ◑ partial.
- **§11 score: ~7.5–8.0, now banked** (Td2-confirmed realized, not projected).

**Remaining work (the new §12 "what's next"):**
1. ✅ done this session (2026-06-27): the `decisions.md` verdict entry + `6_6_e6_T_scaling_draft.tex` v4 are now committed.
2. **§6.6 restructure** (deferred inside v4): add **Mistral as an explicit third base** to `tab:e6-worst` + `tab:e6-slopes`, add the Mistral row to the **Figure 1 cross-model hero**, and add **§6.7 R3 confirmed-prediction language**.
3. **Menu leftovers:** **I** (color-coded heatmap matrix tables), **K** (clean OSS release: README + 1-cmd repro).
4. **Assembly:** P2 abstract/intro/related-work polish → P3 limitations/appendix → P4 anonymize + 300 dpi figs + supplementary zip → compile `main.tex` on Overleaf → submit **ICLR 2027**.

Garg unavailable (family); solo. No arXiv preprint (direct ICLR). Do NOT revert the floor formula (Lemma 2 form), the title/abstract reframe, or weaken the Td2 falsifiability; no GPU 0; no force-push on `phase3-bootstrap`.

*— Claude (Opus 4.8), 2026-06-27.*
