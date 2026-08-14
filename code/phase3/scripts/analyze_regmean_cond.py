"""R1 analyzer. Committed before any cell of the sweep has landed.

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), amended by
notes/prereg_tmlr_amendment_2026-08-14.md (7ce15b1).

  P1  ridge gain G = L(1e-6) - L(lambda*) is larger on the shared arm than on
      the independent arm, on >= 3 of 4 bases.
  P2  the excess at lambda = 1e-6 is worse on the shared arm than on the
      independent arm by a factor of >= 2, on >= 3 of 4 bases.

  CONFIRMED if both, REFUTED if neither, PARTIAL otherwise.

One reading decision is recorded here rather than left implicit. The parent
document states P1 as "larger", with no margin, while its conventions block
carries the 0.005 nat tie threshold over from the earlier registrations, and
E2's analogous prediction used "larger by more than the tie threshold". This
analyzer applies the margin version as PRIMARY, because a bare inequality on a
metric whose own resolution is about 0.001 nats is not a test, and prints the
bare-inequality count alongside so a reader can apply either. If the two
disagree, that is reported in the verdict line rather than resolved silently.

Refuses to emit a verdict on incomplete data.
"""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path.home() / "projects" / "rdmerge"))
R = ROOT / "results" / "phase3" / "eval_regmean_cond"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
LAMBDAS = [(1e-6, "1em6"), (0.01, "0p01"), (0.03, "0p03"), (0.05, "0p05"),
           (0.13, "0p13"), (0.30, "0p30"), (1.00, "1p00")]
COHORTS = [("shared", "seed1"), ("independent", "indep1")]
TIE = 0.005
MIN_RATIO = 2.0


def val(base, tag, cohort):
    p = R / f"{base}__rm_l{tag}__{cohort}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("worst_task_excess")


missing = [f"{b}__rm_l{t}__{c}" for b in BASES for _, t in LAMBDAS
           for _, c in COHORTS if val(b, t, c) is None]
if missing:
    total = len(BASES) * len(LAMBDAS) * len(COHORTS)
    print(f"INCOMPLETE: {len(missing)} of {total} cells missing. "
          f"No verdict will be computed.")
    for m in missing[:10]:
        print("   ", m)
    raise SystemExit(2)

print("=" * 104)
print("R1: RegMean worst-task NLL excess against ridge lambda")
print("    lambda = 0 is not run: rank(sum A^T A) <= T*r = 64 against in_dim")
print("    4096, so it is singular rather than ill-conditioned (amendment 1).")
print("=" * 104)
hdr = "".join(f"{lam:>10}" for lam, _ in LAMBDAS)
print(f"{'base':<13}{'cohort':<14}{hdr}   {'lambda*':>9}{'gain':>9}")
star, gain, floor = {}, {}, {}
for b in BASES:
    for name, c in COHORTS:
        vs = [val(b, t, c) for _, t in LAMBDAS]
        row = "".join(f"{v:>10.4f}" for v in vs)
        i = min(range(len(vs)), key=lambda k: vs[k])
        star[(b, name)] = LAMBDAS[i][0]
        gain[(b, name)] = vs[0] - vs[i]
        floor[(b, name)] = vs[0]
        print(f"{b:<13}{name:<14}{row}   {LAMBDAS[i][0]:>9g}{vs[0]-vs[i]:>9.4f}")
    print()

print("=" * 104)
print(f"P1  ridge gain larger on shared than independent (primary: by > {TIE})")
print("=" * 104)
p1 = p1_bare = 0
for b in BASES:
    gs, gi = gain[(b, "shared")], gain[(b, "independent")]
    ok = (gs - gi) > TIE
    bare = gs > gi
    p1 += ok
    p1_bare += bare
    print(f"{b:<13} gain shared {gs:>9.4f}  independent {gi:>9.4f}"
          f"  diff {gs-gi:>+9.4f} -> {'HOLDS' if ok else 'fails'}"
          f"{'' if ok == bare else '   (bare inequality disagrees)'}")
P1 = p1 >= 3
print(f"\n  holds on {p1}/4 with margin, {p1_bare}/4 bare  ->  "
      f"P1 {'HOLDS' if P1 else 'FAILS'}")

print()
print("=" * 104)
print(f"P2  excess at lambda = 1e-6 worse on shared by a factor >= {MIN_RATIO}")
print("=" * 104)
p2 = 0
for b in BASES:
    fs, fi = floor[(b, "shared")], floor[(b, "independent")]
    ratio = fs / fi if fi > 0 else float("inf")
    ok = ratio >= MIN_RATIO
    p2 += ok
    print(f"{b:<13} shared {fs:>9.4f}  independent {fi:>9.4f}"
          f"  ratio {ratio:>8.2f}x -> {'HOLDS' if ok else 'fails'}")
P2 = p2 >= 3
print(f"\n  holds on {p2}/4  ->  P2 {'HOLDS' if P2 else 'FAILS'}")

verdict = ("CONFIRMED" if (P1 and P2) else
           "REFUTED" if not (P1 or P2) else "PARTIAL")
print()
print("=" * 104)
print(f"R1 VERDICT: {verdict}")
print("=" * 104)
if verdict == "CONFIRMED":
    print("  The conditioning effect appears on a solver we did not build. The")
    print("  family-level claim stands and is reported as tested on two")
    print("  solvers, one of them not ours.")
elif verdict == "PARTIAL":
    print("  Claim ONLY the half that replicated, in the abstract, section 1,")
    print("  section 7.2 and section 8. Cut section 2's family sentence back to")
    print("  the property that actually replicated.")
else:
    print("  The effect did NOT replicate on the one solver we did not build.")
    print("  Per the pre-registration the conditioning claim is narrowed to our")
    print("  own encoder EVERYWHERE: abstract, section 1, section 7.2, section")
    print("  8, and the section 2 sentence naming RegMean, LoRM and RegMean++.")
    print("  The paper states that it did not replicate and that its operational")
    print("  content is correspondingly reduced. Not a footnote, not an appendix.")

print()
print("SCOPE, stated because the registration named it in advance:")
print("  RegMean's minimally regularised reference (1e-6) and our encoder's")
print("  lambda = 0 are not the same object, because the two methods solve in")
print("  different spaces. Each method is compared against ITSELF across arms,")
print("  which is what both predictions are about. No cross-method comparison")
print("  of the two lambda columns is licensed by this design.")
print("  The shared arm is one cohort (seed1), so no across-cohort SE exists")
print("  here either; only the 0.005 nat threshold and the factor-2 rule apply.")
