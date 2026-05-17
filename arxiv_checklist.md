# arXiv v1 pre-launch checklist

*For `arxiv/v1/` bundle. Paper: "A Rate-Distortion Lower Bound
for Model Merging". Goal: post an "embarrassment-free" v1 within
~1 week to establish priority before someone else publishes the
merging-as-rate-distortion framing.*

*Myth check: arXiv does NOT void an ICLR 2027 submission. Dual
peer-review submission is prohibited; preprints are explicitly
allowed. Post freely.*

---

## A. Content blockers (must fix before upload)

### A1. Abstract (empty right now)
- [ ] Write `arxiv/v1/sections/abstract.tex`
- Structure: motivation → RD question → result
  ($\Theta(2^{-2R/(Tr)})$ in the generic regime) → so what.
- 150–200 words. Template in `target.md` §5.
- Write last, after re-reading the body.

### A2. Experiments section (two empty subsection headers right now)
- [ ] Write the synthetic half (`arxiv/v1/sections/experiments.tex`
  §6.1). Three paragraphs:
  1. Slope $-2R/\deff$ across $T\in\{2,3,4,8\}$,
     $r\in\{4,8,16\}$, $d\in\{128,512,2048\}$.
     Cites `day8_rank_r_sanity.py`, `day10_general_ht.py`,
     `day11_task_dep_Dt.py`. Bootstrap 95% CI $\pm 0.01$.
  2. UB/LB ratio $\approx 11$–$13$ constant across rates.
     Cites `day12_achievability.py`,
     `day13_achievability_fixes.py`.
  3. Shared-V null-space split: slope $-1.60 \pm 0.10$ at
     $c=11.5$, $T=2$. Cites `day14c_fractional_bits.py`,
     `day14g_final_lock.py`, `day15_cheb_general_T.py`.
- [ ] Leave §6.2 ("Real-LLM merging") as a short paragraph:
  "deferred to v2, pending Phase 3 experiments". DO NOT
  fabricate any real-LLM numbers.

### A3. Top-10 BibTeX entries (16 "Authors TBD" placeholders right now)
- [ ] Fix these 10 in `arxiv/v1/references.bib` (and mirror in
  `paper/references.bib`):
  1. `zandieh2025turboquant` — arXiv:2504.19874, check author
     ordering
  2. `ortizjimenez2023disentanglement` — NeurIPS 2023 oral
  3. `ilharco2023task` — ICLR 2023
  4. `stoica2025knots` — ICLR 2025
  5. `yadav2023ties` — NeurIPS 2023
  6. `yu2024dare` — ICML 2024
  7. `systematic2025merging` — arXiv:2511.21437
  8. `panariello2025core` — Core Space paper
  9. `matena2022fisher` — NeurIPS 2022
  10. `hu2022lora` — ICLR 2022 (probably already OK)
- The other 6 placeholders (TSPA, DO-Merging, ARM, TARA, QuIP#,
  ATM, etc.) can stay with `note = {Verify canonical cite at
  submission.}` for v1; fix in v2.

### A4. Retitle (optional but recommended, 5 min)
Current title says "Hadamard Incoherence" but the algorithm uses
Gaussian-QR random orthogonal rotation. Options:
- [ ] "A Rate-Distortion Function for Model Merging" (my pick —
  crisp, ML-facing)
- [ ] "Rate-Distortion Limits of Model Merging"
- [ ] "A Rate-Distortion Lower Bound for Model Merging, with
  Matching Achievability via Random Orthogonal Mixing"
  (minimal change, fixes the Hadamard overcommit)

---

## B. Submission mechanics

### B1. arXiv endorsement (blocker, runs in parallel)
First-time submitters to `cs.LG`/`cs.IT` need endorsement from an
existing arXiv author. Takes 1–3 days calendar time.
- [ ] Email Prof. Sanjay Garg: "I'm posting my RD-for-model-
  merging preprint to arXiv this week — would you endorse me for
  cs.LG?" Include the PDF as attachment.
- [ ] If no reply from Garg in 2 days, re-ping Zandieh / Mirrokni
  (pending since 2026-04-26) with the same ask.

### B2. arXiv metadata (fill on the upload form)
- [ ] **Title**: per A4 above.
- [ ] **Authors**: Sankalp Pathak
- [ ] **Email**: pathaksankalp04@gmail.com
- [ ] **Abstract**: per A1 above, paste into the form (arXiv
  strips LaTeX).
- [ ] **Primary category**: `cs.LG`
- [ ] **Cross-list**: `cs.IT`, `stat.ML`
- [ ] **Comments field**: "15 pages. Theory and synthetic
  experiments; real-LLM experiments deferred to v2." Sets
  expectations honestly.

### B3. Final compile + bundle
- [ ] Re-compile `arxiv/v1/main.tex` on Overleaf after A1–A4.
  Download the PDF — that's what reviewers first click.
- [ ] arXiv runs its own pdfLaTeX on upload. Bundle must compile
  cold. Overleaf success is necessary but not sufficient.
- [ ] Zip `arxiv/v1/` or upload via arXiv's tarball flow. Must
  contain: `main.tex`, `references.bib`, `sections/*.tex`.

### B4. Post-upload housekeeping (15 min)
- [ ] Save the assigned arXiv ID (format ~`2604.XXXXX`)
- [ ] Add arXiv URL to `preprint_repo/README.md` on the K144U
  GitHub repo
- [ ] Update `notes/feedback_received.md` with the canonical
  preprint link
- [ ] Verify on `arxiv.org/abs/<id>` that the PDF renders and
  the email isn't mangled in the author listing
- [ ] Sanity: `pdftotext main.pdf - | head -20` should show
  title, author, email, and first abstract line cleanly

---

## C. Do NOT do for v1

- Do **not** fabricate real-LLM numbers in §6.2.
- Do **not** tighten the $C = Tc^2/3$ constant in Thm 9 (v2
  refinement at best).
- Do **not** attempt to close the shared-V exponent gap —
  Phase 2.5 already ruled out the natural LB sharpening.
- Do **not** fix all 16 "Authors TBD" — prioritize the 10 that
  carry argumentative weight.
- Do **not** change the documentclass to ICLR/NeurIPS style.
  arXiv v1 stays plain `article`; change when submitting to the
  target venue.

---

## D. Time estimate

| Item | Effort | Calendar |
|---|---|---|
| A1 Abstract | 30–45 min | same day |
| A2 Experiments (synthetic) | 1 hr | same day |
| A3 Top-10 BibTeX | 1–2 hr (web lookups) | same day |
| A4 Retitle | 5 min | same day |
| B1 Endorsement | 15 min to send; 1–3 days to receive | parallel |
| B2+B3+B4 Upload | 1 hr | day of upload |

Total active work: **4–5 hours**. Calendar: **~1 week** gated by
endorsement.

---

## E. Done when

- arXiv listing live at `arxiv.org/abs/2604.XXXXX`
- PDF renders cleanly on arxiv.org
- Categories: `cs.LG` (primary), `cs.IT` + `stat.ML` (cross-list)
- Email correct on title page
- `preprint_repo/README.md` updated with arXiv link
