"""Is nll_tau bitwise identical across cells that share a (base, cohort)?

If it is, caching it is provably inert: the cache returns exactly what a fresh
run would have computed. If it is not, caching CHANGES the numbers, and the
size of the change is the thing to report.

eval_ridge_cond is the natural test set: 7 lambda cells per (base, cohort),
each of which recomputed the same 20 adapter evaluations independently.
"""
import json
from collections import defaultdict
from pathlib import Path

R = Path(r"D:\my research papers\Clusterexecuted\RDMerge\rdmerge\results\phase3\eval_ridge_cond")

groups = defaultdict(list)
for p in sorted(R.glob("*.json")):
    base, _method, cohort = p.stem.split("__")
    groups[(base, cohort)].append(p)

worst = 0.0
worst_where = None
identical = 0
compared = 0
for (base, cohort), paths in sorted(groups.items()):
    blobs = [json.loads(p.read_text()) for p in paths]
    ref = blobs[0]["nll_tau"]
    same = True
    for b in blobs[1:]:
        for state in ref:
            for task, v in ref[state].items():
                other = b["nll_tau"][state][task]
                compared += 1
                d = abs(other - v)
                if d > worst:
                    worst, worst_where = d, f"{base}/{cohort}/{state}/{task}"
                if other != v:
                    same = False
    identical += same
    print(f"{base:<13}{cohort:<9}{len(paths)} cells  "
          f"{'bitwise identical' if same else 'DIFFERS'}")

print(f"\n{identical} of {len(groups)} (base, cohort) groups bitwise identical")
print(f"{compared} value comparisons, worst absolute difference {worst:.3e}")
if worst_where:
    print(f"worst at {worst_where}")
