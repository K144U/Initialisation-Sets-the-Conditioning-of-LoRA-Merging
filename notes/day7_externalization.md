# Day 7 externalization drafts — 2026-04-26

Per `plan.md` §2.1 Day 7 and `daily_log.md:438-449`. Review, edit, copy-paste.
Each section is self-contained so they can be sent independently.

---

## 0. Pre-send checklist

- [ ] `theory/toy_theorem_v0.tex` compiles cleanly on Overleaf (install
      MiKTeX locally or push the file to Overleaf; resolve any `??`
      broken refs). Target: send PDF attachment alongside the `.tex`.
- [ ] One-shot re-run of `code/synthetic/day5_lower_bound_sanity.py` to
      refresh the numerics. Already confirmed reproducing the Day-5
      addendum ranges on 2026-04-26.
- [ ] Post-send: log sends in `notes/feedback_received.md` (create if
      missing) — who, when, what ask.

---

## 1. Email to the TurboQuant authors (Zandieh, Daliri, Hadian, Mirrokni) — highest stakes

**To:** zandieh@google.com, mirrokni@google.com  
**Cc:** daliri.majid@nyu.edu, majidh@google.com  
**Subject:** Rate-distortion LB for LoRA merging, as a reduction to Thm 3 (6-page note, seeking a 10-min sanity check)

**Attachment:** `rdmerge.pdf` (the compiled `toy_theorem_v0.tex`)

---

Dear Drs. Zandieh, Daliri, Hadian, and Mirrokni,

I'm Sankalp Pathak, a student pursuing a master's in computer science
engineering. The attached 6-page note proves a Shannon-style
rate–distortion lower bound for merging
$T$ LoRA task vectors under a max-over-tasks distortion, as a
reduction to your TurboQuant Theorem 3:
$$\mathcal{D}^\star(T,d,B,R)\;\geq\;B^2\!\left(1-\tfrac{1}{T}\right)
\;+\;c_{\mathrm{TQ}}\cdot\tfrac{B^2}{T}\cdot 2^{-2R/d}.$$
The first term is an irreducible Chebyshev-radius floor; the second
is your Theorem 3 applied to the task-vector centroid via a
scale-invariant RD lemma for rotationally-invariant sources. A
deterministic $\max\geq\mathrm{avg}$ identity (the cross-term
$\sum_t(\tau_t-\bar\tau)=0$ identically) is what glues the two. At
$T=1$ the bound recovers Theorem 3 with identical constant, and a
quantized-Chebyshev-center construction matches it up to constants
numerically.

Would any of you have 10 minutes in the next 1–2 weeks to tell me
whether this reduction framing is the one you had in mind, or whether
I'm missing something? The one step I'd most value an outside read on
is the application of the scale-invariant lemma to the centroid
$\bar\tau$: there is no deterministic lower bound on $\|\bar\tau\|^2$
(at $T=2$, $\tau_1=-\tau_2$ gives $\|\bar\tau\|=0$), and I currently
patch this with a high-probability norm event and absorb a $(1+o_d(1))$
prefactor into $c_{\mathrm{TQ}}$. An EPI-style rewrite that bypasses
the radius-conditioning would be cleaner, but I don't yet see the
right entropy object.

This is solo work targeted at ICLR 2027; you're the closest audience
to sanity-check the RD half before I broaden the ask. A short
reaction — even "that's what we intended", "you missed reference X",
or "this is wrong for reason Y" — would be useful. Happy to send the
LaTeX source, the numerics script, or a brief walkthrough if easier.

Thank you for your time, and congratulations on TurboQuant.

Best,

Sankalp Pathak  
pathaksankalp04@gmail.com

---

## 2. Alignment Forum research note

**Title:** Rate-distortion lower bound for LoRA model merging (research note, seeking feedback)  
**Tags:** AI, Information Theory, Machine Learning  
**Post type:** Research note / Draft  
**Target:** shorter, informal, aimed at "1 outside reader engages by Week 2" per plan.md §10.

---

**TL;DR.** I'm a solo researcher working on a theoretical paper about
LoRA merging: *how many bits do you need to store a merged model that
preserves $\epsilon$-distortion on each of $T$ tasks?* I have a
2-page sketch of a Shannon-style lower bound that reduces to
[TurboQuant Theorem 3](https://arxiv.org/abs/2504.19874) via a
deterministic $\max\geq\mathrm{avg}$ identity plus a rotationally-
invariant rate–distortion lemma on the task-vector centroid. Looking
for a sanity check on the reduction and on a Chebyshev-based
tail-event argument in Lemma 2.

## Setting

$T$ task vectors $\tau_1,\dots,\tau_T\in\mathbb{R}^d$ with
$\|\tau_t\|\leq B$. Per-task quadratic loss $f_t(w)=\tfrac12\|w-\tau_t\|^2$.
A rate-$R$ merging algorithm is an encoder–decoder pair
$E:(\mathbb{R}^d)^T\to\{1,\dots,2^R\}$, $D:\{1,\dots,2^R\}\to\mathbb{R}^d$
producing $w^\star=D(E(\boldsymbol\tau))$. Figure of merit: max-distortion
$\Delta(w^\star;\boldsymbol\tau)=\max_t\|w^\star-\tau_t\|^2$.

## Result

$$\mathcal{D}^\star(T,d,B,R)\;\geq\;B^2\!\left(1-\tfrac1T\right)
\;+\;c_{\mathrm{TQ}}\cdot\tfrac{B^2}{T}\cdot 2^{-2R/d}.$$

Two terms with distinct meanings:
- $B^2(1-1/T)$ is a **Chebyshev-radius floor** — the irreducible
  multi-task incompatibility. Exactly $\mathbb{E}[(1/T)\sum_t\|\tau_t-\bar\tau\|^2]$
  under iid uniform-sphere task vectors; numerically within 1–2% of
  $\mathbb{E}[R_c^2]$.
- $c_{\mathrm{TQ}}(B^2/T)\cdot 2^{-2R/d}$ is the classical Shannon
  decay governed by the rate–distortion function of the centroid
  $\bar\tau$.

## Proof in one paragraph

Yao's minimax with hard distribution $P=\mathrm{Unif}(BS^{d-1})^{\otimes T}$.
The deterministic identity
$$\tfrac1T\sum_t\|w^\star-\tau_t\|^2=\|w^\star-\bar\tau\|^2+\tfrac1T\sum_t\|\tau_t-\bar\tau\|^2$$
holds because the cross-term $\sum_t(\tau_t-\bar\tau)=0$ **identically**.
Take max on the LHS and expectations: $\mathbb{E}[\max_t\|w^\star-\tau_t\|^2]
\geq\mathbb{E}\|w^\star-\bar\tau\|^2 + B^2(1-1/T)$. Now apply a
rate–distortion lower bound to the first term: under $P$, $\bar\tau$
is rotationally invariant with $\mathbb{E}\|\bar\tau\|^2=B^2/T$, so the
scale-invariant bound $\mathbb{E}[\|\hat Y-Y\|^2/\|Y\|^2]\geq
c_{\mathrm{TQ}}\cdot 2^{-2R/d}$ from TurboQuant Thm 3 transfers
(chain rule + DPI + Jensen). Converting to absolute scale uses a
high-probability norm-event $G_d$ with $\Pr(G_d^c)=O(d^{-1})$,
absorbing a $(1+o_d(1))$ factor into the constant.

## Novelty claim

Standard multi-source rate–distortion frameworks (multiple descriptions,
Heegard-Berger, common reconstruction) all use **per-source** or
**average** distortion; none treat $\max_t D_t$ as the figure of merit.
The $\max$ is precisely what isolates the multi-task Chebyshev floor,
and the $\max\geq\mathrm{avg}$ reduction is what separates this floor
from the compression decay. I believe this is the first clean
rate–distortion statement for max-distortion multi-source with a
single-vector-RD reduction.

## Numerics

$T\in\{2,4,8\}$, $d\in\{128,512,2048\}$, $b=R/d\in\{1,2,3,4,6,8,12\}$,
30 trials. Three checks:

- **Chebyshev-floor formula $B^2(1-1/T)$.** Empirical/theory ratio in
  $[0.9984, 1.0173]$ across all $(T,d)$.
- **LB respected.** Valid everywhere (as it must be).
- **$2^{-R/d}$ regime — resolved NO.** A quantized-Chebyshev-center
  merge closes the 2-regime apparent decay; the $2^{-R/d}$ behavior
  in an earlier mean-merge experiment was a sub-optimal-centering
  plateau, not an intrinsic RD phenomenon. The rate-distortion
  function is $\Theta\bigl(R_c^2+(B^2/T)\cdot 2^{-2R/d}\bigr)$.

## What I'm asking

Two specific checks:
1. **Lemma 2's tail-event substitution.** $Y=\bar\tau$ has no
   deterministic lower bound on $\|Y\|^2$ ($T=2$, $\tau_1=-\tau_2$
   kills it). I use $G_d=\{\|\bar\tau\|^2\geq B^2/(T\kappa_d)\}$ with
   $\Pr(G_d^c)$ controlled by $\mathrm{Var}(\|\bar\tau\|^2)=O(B^4/(T^2d))$.
   Is there a reason to prefer an EPI-based derivation that bypasses
   the $\rho$-conditioning entirely?
2. **Novelty.** I checked El Gamal-Cover 1982, Steinberg 2009,
   Heegard-Berger, Wyner-Ziv; none match. Am I missing an earlier
   treatment of max-over-T distortion that I should be citing?

Any other pointers — "you missed reference $X$", "Lemma 2 is cleaner
if you do $Y$", "this is subsumed by $Z$" — very welcome. Full 6-page
note in comments / first reply (or DM me if preferred).

---

## 3. Prof. Sanjay Garg DM / email — 2-line ping

**Channel:** whichever you normally use (LinkedIn DM / email)

---

Hi Prof. Garg —

Quick Phase 0 update on the LoRA-merging rate–distortion paper: the
toy theorem closed this week as a reduction to TurboQuant Thm 3 + a
deterministic max$\geq$avg identity. Decision-gate row 1 (proceed to
Phase 1). If you have 20 minutes this week I'd love to walk you
through the proof and get your read on the novelty framing before I
cold-email the TurboQuant authors. Happy to send the 2-page note
first if that's easier.

Thanks,
Sankalp

---

## Sending order (suggested)

1. **Garg first** (low-stakes, short DM). Gives a friendly reader
   before the cold email goes out. If he catches anything, fix and
   resend to Zandieh/Mirrokni.
2. **Zandieh/Mirrokni email** next, once the .tex compiles on
   Overleaf and you have a PDF attachment.
3. **Alignment Forum post** last, 24-48 hours after the email — if
   Zandieh responds fast, their reply may shape the AF post's framing.

## Things to do at the same time as sending

- Create `notes/feedback_received.md` to log everything that comes
  back.
- Post the note to a public GitHub gist (or arXiv-endorsed preprint
  if you have an endorser) so the Alignment Forum post can link to a
  stable URL rather than copy-pasting the full proof.
