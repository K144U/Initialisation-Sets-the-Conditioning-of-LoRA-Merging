"""E2 analyzer. Rules fixed in notes/prereg_conditioning_2026-08-07.md (f9d230e),
committed before the cells were generated or dispatched.

P1: lambda*(shared) >= 10 x lambda*(independent) on >= 3 of 4 bases. A
    lambda* of 0 on the independent arm satisfies the inequality for any
    non-zero shared lambda*.
P2: ridge gain G = L(lambda=0) - L(lambda*) is larger on the shared arm than
    the independent arm by more than 0.005 nats on >= 3 of 4 bases.

CONFIRMED if both, PARTIAL if exactly one, REFUTED if neither. On REFUTED the
refutation is the headline and is not demoted to a limitation (constraint 5).

Refuses to emit a verdict on incomplete data: a previous analyzer in this
project printed a full verdict off an empty directory.
"""
import json
from pathlib import Path

R = Path.home() / "projects" / "rdmerge" / "results" / "phase3" / "eval_ridge_cond"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
LAMBDAS = [(0.0, "0"), (0.01, "0p01"), (0.03, "0p03"), (0.05, "0p05"),
           (0.13, "0p13"), (0.30, "0p30"), (1.00, "1p00")]
COHORTS = [("shared", "seed1"), ("independent", "indep1")]
TIE = 0.005


def val(base, tag, cohort):
    p = R / f"{base}__rd_l{tag}__{cohort}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("worst_task_excess")


# ---------- completeness gate ----------
missing = [f"{b}__rd_l{t}__{c}" for b in BASES for _, t in LAMBDAS
           for _, c in COHORTS if val(b, t, c) is None]
if missing:
    print(f"INCOMPLETE: {len(missing)} of {len(BASES)*len(LAMBDAS)*len(COHORTS)} "
          f"cells missing. No verdict will be computed.")
    for m in missing[:10]:
        print("   ", m)
    raise SystemExit(2)

# ---------- full curves (prereg constraint 4: report every lambda) ----------
print("=" * 100)
print("E2: worst-task NLL excess against ridge lambda, rank pinned to 16")
print("=" * 100)
hdr = "".join(f"{lam:>9}" for lam, _ in LAMBDAS)
print(f"{'base':<13}{'cohort':<14}{hdr}   {'lambda*':>9}{'gain':>9}")
star, gain = {}, {}
for b in BASES:
    for name, c in COHORTS:
        vs = [val(b, t, c) for _, t in LAMBDAS]
        row = "".join(f"{v:>9.4f}" for v in vs)
        i = min(range(len(vs)), key=lambda k: vs[k])
        star[(b, name)] = LAMBDAS[i][0]
        gain[(b, name)] = vs[0] - vs[i]
        print(f"{b:<13}{name:<14}{row}   {LAMBDAS[i][0]:>9}{vs[0]-vs[i]:>9.4f}")
    print()

# ---------- P1 ----------
print("=" * 100)
print("P1  lambda*(shared) >= 10 x lambda*(independent)")
print("=" * 100)
p1 = 0
for b in BASES:
    s, i = star[(b, "shared")], star[(b, "independent")]
    ok = (s > 0 and i == 0) or (i > 0 and s >= 10 * i)
    p1 += ok
    print(f"{b:<13} shared {s:<8} independent {i:<8} -> {'HOLDS' if ok else 'fails'}")
P1 = p1 >= 3
print(f"\n  holds on {p1}/4  ->  P1 {'HOLDS' if P1 else 'FAILS'}")

# ---------- P2 ----------
print()
print("=" * 100)
print("P2  ridge gain larger on shared than independent by > 0.005 nats")
print("=" * 100)
p2 = 0
for b in BASES:
    gs, gi = gain[(b, "shared")], gain[(b, "independent")]
    ok = (gs - gi) > TIE
    p2 += ok
    print(f"{b:<13} gain shared {gs:>8.4f}  independent {gi:>8.4f}"
          f"  diff {gs-gi:>+8.4f} -> {'HOLDS' if ok else 'fails'}")
P2 = p2 >= 3
print(f"\n  holds on {p2}/4  ->  P2 {'HOLDS' if P2 else 'FAILS'}")

# ---------- verdict ----------
verdict = ("CONFIRMED" if (P1 and P2) else
           "PARTIAL" if (P1 or P2) else "REFUTED")
print()
print("=" * 100)
print(f"E2 VERDICT: {verdict}")
print("=" * 100)
if verdict == "CONFIRMED":
    print("  Conditioning has an operational consequence. It explains Q1':")
    print("  the encoder's ridge was repairing conditioning, and bought nothing")
    print("  where there was nothing to repair. This becomes the paper's spine.")
elif verdict == "PARTIAL":
    print("  Report both. Claim ONLY the prediction that held. The one that")
    print("  failed is not narrated as a trend (prereg decision rules).")
else:
    print("  Conditioning differs by four orders of magnitude and has NO")
    print("  operational consequence for any method we can test. Per prereg")
    print("  constraint 5 this is the HEADLINE, written as the result. It is")
    print("  not demoted to a limitation and the spine is not quietly swapped.")

print()
print("LIMITATION, stated because the prereg named a gate we cannot compute:")
print("  the pre-registration specified a one-directional 2 x SE noise gate.")
print("  This sweep has ONE cohort per arm (seed1, indep1), so there is no")
print("  across-cohort SE to compute. Only the 0.005 nat threshold is applied.")
print("  Differences near that threshold are correspondingly less certain.")
