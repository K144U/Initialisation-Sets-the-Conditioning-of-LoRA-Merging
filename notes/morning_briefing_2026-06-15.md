# Morning briefing — 2026-06-15

You went to sleep around 19:55 IST on 2026-06-14, after launching E3
GSM8K em sweep (20 cells). Here's what to check when you wake up.

---

## First 30 seconds

1. **Check your phone** for the cron's PushNotification with the 4×5
   accuracy table. Expected delivery: ~05:50 IST.
2. If no ping → check `qstat -u sanjay.g`. If `rdm_e3gsm` is still
   running, the job is just slow. Cron will ping when it finishes.
3. If a ping that says something other than "P5 E3 GSM8K sweep done":
   see the *if something broke* section.

## First 5 minutes

Run the analysis script to see the verdict:

```bash
PYTHONNOUSERSITE=1 ~/projects/rdmerge/.conda/envs/rdmerge/bin/python \
    ~/projects/rdmerge/code/phase3/scripts/analyze_e3_gsm8k.py
```

This prints:
- 4×5 accuracy table
- 4×5 NLL-excess table (from §6.1 matrix, for comparison)
- Per-model Spearman correlation (accuracy ↔ −NLL excess)
- Best method per model
- Decision-rule verdict (STRONG / MODERATE / WEAK per model)

Output also written to `results/phase3/e3_gsm8k_summary.json`.

---

## What the data will tell us

### Decision rule (master plan §E3)

| Per-model Spearman ρ | Implication | Paper consequence |
|---|---|---|
| ≥ 0.7 in all 4 | NLL conclusions hold on accuracy | §6.5 = confirmation paragraph |
| 0.4 ≤ ρ < 0.7 | Moderate agreement | §6.5 = "noisier but consistent" |
| < 0.4 | NLL not predictive | §6.5 = separate story, needs care |

### Reference baseline

The smoke (qwen + TIES + GSM8K, n=100) hit 0.78 accuracy. For the
sweep (n=500), expect numbers in the **0.30–0.85** range across cells.
The qwen + TIES cell will likely land near 0.78 (validating the smoke).

### What to expect per model

From §6.1 NLL rankings, the order should be roughly (best → worst):
- llama: TIES → b2 → (TA, DARE, KnOTS, b4) cluster
- mistral: TIES ≈ b2 → (others) cluster
- qwen: TIES ≈ b2 → (others)
- yi: TIES → (others, including b2)

If accuracy follows this same order per model, ρ will be high → strong agreement.

---

## If something broke

### Walltime hit before 20 cells finished

Cron should auto-requeue (`qsub pbs_orchestrator_e3_gsm8k.sh`). Check:

```bash
qstat -u sanjay.g                                          # is rdm_e3gsm running again?
ls ~/projects/rdmerge/results/phase3/eval_e3_gsm8k/ | wc -l  # how many landed
```

If it's queued but not running, just let it run — orchestrator skips
done files.

### Some cells failed

```bash
ls ~/projects/rdmerge/logs/orch/e3g_*.log
grep -l "Error\|Traceback" ~/projects/rdmerge/logs/orch/e3g_*.log
```

Most likely cause: a model file is missing or one cell crashed on a
specific config. Look at the log for that cell, fix the config, requeue.

### Pipeline crashed entirely (no JSONs landed)

Read the spool: `ssh jiit-gpu01 'cat /var/spool/pbs/spool/41578.*.OU | tail -50'`.
Most likely cause: missing dependency, GPU eviction, or VRAM
exhaustion. Diagnose and resubmit the wrapper.

---

## What's ready to ship next (after E3 GSM8K lands)

| Priority | Item | Cost | Why now |
|---|---|---|---|
| 1 | Write §6.5 (E3) paragraph using the verdict | ~30 min | Lock in the result while fresh |
| 2 | Decide: scale to n=1000 OR add HumanEval pass@1 | – | Your call based on Spearman result |
| 3 | Update decisions.md entry 17 | ~5 min | Routine |
| 4 | E6 (T sweep on real adapters) — start config gen | ~half day | The next big-ticket item |

---

## Repo state at sleep time

- Branch `phase3-bootstrap`, latest commit `3605ef5` (E3 GSM8K sweep launched)
- Working tree clean
- Cron `dc10ce97` ticking
- No uncommitted work; you can pull and review freshly

---

## Pinging me when you're up

Just describe your priority. If accuracy looks great → "let's write §6.5
and pick the next experiment." If something looks off → "the qwen TVQ
b=2 cell hit 0.10, what gives?" Either way I'll pick up where this
left off.

Good morning when you read this.
