"""W4: the head-to-head the reviewer says is missing.

Worst-task NLL excess, same method, same base, shared-initialisation cohort
(seed1/2/3) against independently initialised cohort (indep1/2/3).

The paper's practical recommendation is "initialise independently". That
recommendation is only supported if merging actually goes better there. This
script reports whether it does, in whichever direction the data falls.

IMPORTANT CONFOUND, stated up front: the shared arm varies `seeds.global`,
which drives BOTH the init draw and the training-data shuffle, while the
independent arm varies the init draw only with the data seed pinned. The
arms are therefore not a clean A/B on initialisation alone. This script also
checks whether the eval-side configuration matches between the arms and says
so, because if it does not, the comparison is not licensed at all.
"""
import json
import statistics
from pathlib import Path

R = Path.home() / "projects" / "rdmerge" / "results" / "phase3"
SHARED_DIR, INDEP_DIR = R / "eval_matrix_seeds", R / "eval_a1_indep"
SHARED = ["seed1", "seed2", "seed3"]
INDEP = ["indep1", "indep2", "indep3"]
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]


def load(d, base, method, cohort):
    p = d / f"{base}__{method}__{cohort}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def excess(d, base, method, cohort):
    j = load(d, base, method, cohort)
    return None if j is None else j.get("worst_task_excess")


def methods_present():
    m = set()
    for p in INDEP_DIR.glob("*__indep1.json"):
        m.add(p.name.split("__")[1])
    m2 = set()
    for p in SHARED_DIR.glob("*__seed1.json"):
        m2.add(p.name.split("__")[1])
    return sorted(m & m2)


def cfgkey(j):
    c = (j or {}).get("meta", {}).get("config", {})
    return (c.get("base_model"), c.get("max_seq_length"), c.get("seed"),
            tuple(s.get("task_cfg", {}).get("n_eval")
                  for s in c.get("adapter_specs", [])))


METHODS = methods_present()
print("methods in both arms:", METHODS)
print()

# --- eval-config equality check -------------------------------------------
print("=" * 78)
print("EVAL CONFIG EQUALITY between the two arms (must match to compare)")
print("=" * 78)
mismatch = 0
for b in BASES:
    a = cfgkey(load(SHARED_DIR, b, METHODS[0], "seed1"))
    c = cfgkey(load(INDEP_DIR, b, METHODS[0], "indep1"))
    ok = a == c
    mismatch += (not ok)
    print(f"  {b:<12} {'MATCH' if ok else 'DIFFER'}")
    if not ok:
        print(f"     shared: {a}")
        print(f"     indep : {c}")
print()

# --- the head-to-head ------------------------------------------------------
print("=" * 78)
print("WORST-TASK NLL EXCESS: shared init vs independent init")
print("negative delta = independent init is WORSE (higher excess)")
print("=" * 78)
print(f"{'base':<12}{'method':<16}{'shared':>9}{'indep':>9}{'delta':>10}  {'':<6}")
wins = losses = ties = 0
rows = []
for b in BASES:
    for m in METHODS:
        sv = [excess(SHARED_DIR, b, m, c) for c in SHARED]
        iv = [excess(INDEP_DIR, b, m, c) for c in INDEP]
        sv = [v for v in sv if v is not None]
        iv = [v for v in iv if v is not None]
        if len(sv) < 2 or len(iv) < 2:
            continue
        s, i = statistics.fmean(sv), statistics.fmean(iv)
        d = s - i  # positive => independent is better (lower excess)
        tag = "indep better" if d > 0.005 else (
            "indep WORSE" if d < -0.005 else "tie")
        wins += d > 0.005
        losses += d < -0.005
        ties += abs(d) <= 0.005
        rows.append((b, m, s, i, d))
        print(f"{b:<12}{m:<16}{s:9.4f}{i:9.4f}{d:+10.4f}  {tag}")
    print()

print("=" * 78)
print(f"independent better on {wins}, tie on {ties}, WORSE on {losses}"
      f"   (of {wins+ties+losses} base x method cells)")
if rows:
    md = statistics.fmean([r[4] for r in rows])
    print(f"mean delta across all cells: {md:+.4f} nats"
          f"  ({'independent better' if md > 0 else 'independent WORSE'})")
print("=" * 78)
print()
print("CONFOUND, restated: the shared arm varies init AND data shuffle; the")
print("independent arm varies init only. Read this as the head-to-head the")
print("recommendation implies, not as a clean isolation of initialisation.")
if mismatch:
    print()
    print("WARNING: eval configs DIFFER between arms on some base. The")
    print("comparison above is NOT licensed until that is resolved.")
