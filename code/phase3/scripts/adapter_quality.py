"""E3 (no-GPU half): did the adapters actually learn their tasks?

An external review (W9) points out that we never report per-adapter task
performance, so a reader cannot tell whether the cohorts are undertrained. If
they are, the near-collinearity under shared initialisation is guaranteed by
construction rather than being a property worth reporting, and every geometry
claim in the paper has to be scoped.

No new compute is needed: every eval cell already stores the full nll_tau
matrix, i.e. the NLL of the base model and of each task adapter on every task.

Two things are measured per (base, cohort, task):
  1. improvement = nll(base) - nll(specialist), in nats and as a fraction.
  2. whether the specialist is actually the BEST adapter on its own task. If
     some other task's adapter beats it, that adapter did not learn a
     distinguishing skill and the cohort is not doing what the paper assumes.
"""
import json
import statistics
from pathlib import Path

R = Path.home() / "projects" / "rdmerge" / "results" / "phase3"
ARMS = [("shared", R / "eval_matrix_seeds", ["seed1", "seed2", "seed3"]),
        ("independent", R / "eval_a1_indep", ["indep1", "indep2", "indep3"])]
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
PROBE = "ties"   # any method works; nll_tau does not depend on the merge


def cell(d, base, cohort):
    p = d / f"{base}__{PROBE}__{cohort}.json"
    return json.loads(p.read_text()) if p.exists() else None


print("=" * 96)
print("PER-ADAPTER TASK QUALITY (from nll_tau; no new compute)")
print("=" * 96)
usurped_all = []
for arm, d, cohorts in ARMS:
    print(f"\n### {arm}")
    print(f"{'base':<12}{'task':<13}{'base NLL':>10}{'spec NLL':>10}"
          f"{'gain':>9}{'gain %':>9}   best adapter on this task")
    for base in BASES:
        rows = [cell(d, base, c) for c in cohorts]
        rows = [r for r in rows if r]
        if not rows:
            print(f"{base:<12} (no cells)")
            continue
        tasks = list(rows[0]["nll_tau"]["base"])
        for t in tasks:
            bn = statistics.fmean(r["nll_tau"]["base"][t] for r in rows)
            sn = statistics.fmean(r["nll_tau"][t][t] for r in rows)
            # which adapter is best on task t, averaged over cohorts
            per_ad = {a: statistics.fmean(r["nll_tau"][a][t] for r in rows)
                      for a in tasks}
            best = min(per_ad, key=per_ad.get)
            flag = "" if best == t else f"  <-- {best} BEATS the specialist"
            if best != t:
                usurped_all.append((arm, base, t, best,
                                    per_ad[t] - per_ad[best]))
            print(f"{base:<12}{t:<13}{bn:>10.4f}{sn:>10.4f}"
                  f"{bn-sn:>9.4f}{100*(bn-sn)/bn:>8.1f}%   {best}{flag}")
    print()

print("=" * 96)
if usurped_all:
    print(f"SPECIALIST NOT BEST ON ITS OWN TASK in {len(usurped_all)} "
          f"(arm, base, task) combinations:")
    for arm, base, t, best, gap in usurped_all:
        print(f"   {arm:<12}{base:<12}{t:<13} beaten by {best} by {gap:.4f} nats")
    print()
    print("Consequence: in these cells the adapter did not learn a skill that")
    print("distinguishes it from its cohort-mates. Undertraining is NOT excluded")
    print("and the geometry claims must be scoped to this training budget.")
else:
    print("Every specialist is the best adapter on its own task, in every arm.")
print("=" * 96)
