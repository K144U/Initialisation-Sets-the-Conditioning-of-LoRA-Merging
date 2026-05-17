# rdmerge — Rate-Distortion Lower Bound for LoRA Model Merging

A Shannon-style rate–distortion lower bound for the problem of
compressing $T$ LoRA-fine-tuned task vectors into a single merged
model, with worst-task distortion as the figure of merit.

The lower bound reduces to
[TurboQuant Theorem 3](https://arxiv.org/abs/2504.19874) (Zandieh,
Daliri, Hadian, Mirrokni — 2504.19874) via a deterministic
max-$\geq$-avg identity on the $T$ task vectors plus a scale-invariant
RD lemma for rotationally-invariant sources.

## Current result (Phase 0, 6-page note)

For $\tau_1, \dots, \tau_T \in \mathbb{R}^d$ with $\|\tau_t\| \leq B$
iid uniform on the sphere, and a rate-$R$ encoder–decoder producing
a merged $w^\star$:

$$
\mathcal{D}^\star(T, d, B, R)
\;\geq\;
B^2\!\left(1 - \tfrac{1}{T}\right)
\;+\; c_{\mathrm{TQ}} \cdot \tfrac{B^2}{T} \cdot 2^{-2R/d}.
$$

The first term is an irreducible Chebyshev-radius floor; the second
is the classical Shannon decay, with $c_{\mathrm{TQ}}$ the universal
constant of TurboQuant Theorem 3. At $T = 1$ the bound recovers
Theorem 3 with identical constant; numerical sanity checks (30 trials
across $T \in \{2, 4, 8\}$, $d \in \{128, 512, 2048\}$) confirm the
floor formula to within 1–2%.

See [`rdmerge.pdf`](rdmerge.pdf) for the full note (6 pages,
self-contained proof with Yao minimax + rotation-invariant Lemma 2
+ one-paragraph numerical verification).

## Status

Work in progress. Phase 0 note (this PDF) closes the simplest
case: isotropic per-task Hessian $H_t = I_d$, sphere-uniform
$\tau_t$. Phase 1 (not yet in this repo) lifts to rank-$r$ LoRA
subspaces with general $H_t \succeq 0$ — a rank-$r$ effective-
dimension parameter $d_{\mathrm{eff}}$ replaces $d$ in the exponent,
and the floor becomes $B^2(1 - d_{\mathrm{eff}}/(Tr))$. Phase 2
will add a matching achievability upper bound via Hadamard-
incoherence quantization. Target venue: ICLR 2027.

## Feedback

Solo research; feedback very welcome. Especially on:

1. **Lemma 2's tail-event substitution.** The note uses a high-
   probability norm event on $\|\bar\tau\|^2$ and absorbs a
   $(1 + o_d(1))$ prefactor into the constant; an entropy-power-
   inequality rewrite that bypasses this conditioning would be
   cleaner but I don't yet see the right entropy object.
2. **Novelty.** The $\max_t D_t$ figure of merit does not appear to
   be covered by standard multi-source RD frameworks (El Gamal–Cover
   1982, Heegard–Berger, Wyner–Ziv, Steinberg 2009). Pointers to
   prior work very welcome.

Open a GitHub issue, or email **pathaksankalp04@gmail.com**.

## Citation

```bibtex
@misc{pathak2026rdmerge,
  author       = {Sankalp Pathak},
  title        = {A Rate-Distortion Lower Bound for LoRA Model Merging
                  (Phase 0 note)},
  year         = {2026},
  note         = {Work in progress},
  howpublished = {\url{https://github.com/K144U/rdmerge-preprint}}
}
```

## License

Released under **Creative Commons Attribution 4.0 International
(CC BY 4.0)** — see [`LICENSE`](LICENSE). You can share, adapt, or
build on this work with attribution; commercial use is allowed.
