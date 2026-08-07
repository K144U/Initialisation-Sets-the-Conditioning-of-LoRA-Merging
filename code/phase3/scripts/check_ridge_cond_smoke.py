"""Smoke checks for E2, per prereg constraint 3.

The two smoke cells are qwen at lambda 0.05 on both cohorts. They must differ
ONLY in the adapter cohort. Anything else differing means the generator's
retarget is wrong and the whole 56-cell sweep would be confounded.

Reads meta.config, not a top-level config key: an earlier smoke checker in this
project read the wrong nesting, compared None to None, and reported four
vacuous passes.
"""
import json
from pathlib import Path

R = Path("/home/sanjay.g/projects/rdmerge/results/phase3/eval_ridge_cond")
SH = R / "qwen25_7b__rd_l0p05__seed1.json"
IN = R / "qwen25_7b__rd_l0p05__indep1.json"

ok = True


def chk(label, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + label + ("   " + detail if detail else ""))
    if not cond:
        ok = False


for p in (SH, IN):
    if not p.exists():
        print("MISSING", p)
        raise SystemExit(2)

sh, ind = json.loads(SH.read_text()), json.loads(IN.read_text())
sc = sh.get("meta", {}).get("config", {})
ic = ind.get("meta", {}).get("config", {})

chk("shared config block non-empty", bool(sc), "%d keys" % len(sc))
chk("indep config block non-empty", bool(ic), "%d keys" % len(ic))

for j, nm in ((sh, "shared"), (ind, "indep")):
    v = j.get("worst_task_excess")
    chk(f"{nm}: worst_task_excess finite",
        isinstance(v, (int, float)) and v == v, repr(v))

# the two knobs the pre-registration pins
for c, nm in ((sc, "shared"), (ic, "indep")):
    kw = c.get("method_kwargs", {})
    chk(f"{nm}: realize pinned to rank_r", kw.get("realize") == "rank_r",
        repr(kw.get("realize")))
    chk(f"{nm}: ridge_lambda 0.05", abs(float(kw.get("ridge_lambda", -1)) - 0.05) < 1e-9,
        repr(kw.get("ridge_lambda")))
    chk(f"{nm}: method is rd_encoder", c.get("method") == "rd_encoder",
        repr(c.get("method")))

# eval side must be IDENTICAL between arms
for k in ("base_model", "max_seq_length", "seed", "min_free_gb"):
    a, b = sc.get(k), ic.get(k)
    chk("eval config identical across arms: " + k,
        a is not None and a == b, "%r vs %r" % (a, b))

ss, iss = sc.get("adapter_specs", []), ic.get("adapter_specs", [])
chk("4 adapter specs both arms", len(ss) == 4 and len(iss) == 4,
    "%d / %d" % (len(ss), len(iss)))
chk("task order identical",
    [s.get("name") for s in ss] == [s.get("name") for s in iss])
chk("n_eval identical",
    [s.get("task_cfg", {}).get("n_eval") for s in ss]
    == [s.get("task_cfg", {}).get("n_eval") for s in iss])

# the ONE thing that must differ: the cohort
chk("shared arm points at seed1",
    all(s["dir"].rstrip("/").endswith("seed1") for s in ss),
    ss[0]["dir"] if ss else "")
chk("indep arm points at indep1",
    all(s["dir"].rstrip("/").endswith("indep1") for s in iss),
    iss[0]["dir"] if iss else "")

v1, v2 = sh.get("worst_task_excess"), ind.get("worst_task_excess")
chk("the two arms actually differ (cohort swap is not a no-op)",
    abs(v1 - v2) > 1e-6,
    "shared %.4f vs indep %.4f, diff %+.4f" % (v1, v2, v1 - v2))

print()
print("SMOKE " + ("PASSED" if ok else "FAILED"))
raise SystemExit(0 if ok else 1)
