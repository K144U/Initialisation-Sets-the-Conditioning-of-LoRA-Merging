# BibTeX entries needing manual verification before arXiv upload

The 10 entries below in `arxiv/v1/references.bib` currently have
`Authors TBD` placeholders because deepresearch.md did not list
canonical author names and my (Claude's) training cutoff predates
several of these papers. **Verify and fix before uploading to arXiv.**

The fix for each entry is mechanical: open the arXiv abs page (or the
venue proceedings page), copy the author list into the `author = {...}`
field, and delete the `note = {Verify canonical cite at submission.}`
line.

## High priority — these carry argumentative weight in the paper

### 1. `zandieh2025turboquant` — TurboQuant
- **arXiv**: 2504.19874 (`https://arxiv.org/abs/2504.19874`)
- **Fix needed**: full author list + canonical title (current title
  guessed from abstract keywords may be wrong).
- **Why it matters**: direct intellectual ancestor. Wrong cite =
  embarrassing.

### 2. `systematic2025merging` — Nov 2025 negative result
- **arXiv**: 2511.21437 (`https://arxiv.org/abs/2511.21437`)
- **Fix needed**: authors + canonical title.
- **Why it matters**: §2 (related work) directly confronts this
  paper's findings.

### 3. `stoica2025knots` — KnOTS
- **Venue**: ICLR 2025.
- **Current**: `Stoica, George and Ramesh, Pratik and Shah, Viraj`
- **Fix needed**: verify against OpenReview — KnOTS has more authors;
  my 3-author list is likely incomplete.

### 4. `panariello2025core` — Core Space
- **Current**: `Panariello, Antonio and others`
- **Fix needed**: search arXiv for the Panariello 2025 Core Space
  paper; get full author list + arXiv ID.

## Medium priority — cited once in related work

### 5. `tspa2025` — TSPA (Stiefel manifold rotation merging)
- **Current**: OpenReview placeholder.
- **Fix needed**: full cite or drop from related work if not findable.

### 6. `domerging2025` — DO-Merging
- **Venue**: NeurIPS 2025.
- **Fix needed**: author list from OpenReview / proceedings.

### 7. `arm2026streaming` — ARM / Streaming merging
- **arXiv**: 2602.03237 (`https://arxiv.org/abs/2602.03237`)
  — NOTE: deepresearch.md says Feb 2026; arXiv ID format suggests
  this may have been misread (2602 is an unusual prefix — check).
- **Fix needed**: verify arXiv ID, get authors.

### 8. `tara2025` — TARA-Merging
- **Fix needed**: author list + arXiv ID.

### 9. `kim2025tvq` — Task-Vector Quantization
- **Venue**: ICCV 2025.
- **Current**: `Kim, {Authors TBD}`
- **Fix needed**: verify Kim is the first author, add full list.

### 10. `bit1merging2025` — 1-bit-Merging
- **Fix needed**: author list + arXiv ID.

## Already canonical — do NOT need changes

These compiled correctly and will render cleanly:
- `hu2022lora` (LoRA, ICLR 2022)
- `ilharco2023task` (Task Arithmetic, ICLR 2023)
- `yadav2023ties` (TIES, NeurIPS 2023)
- `yu2024dare` (DARE, ICML 2024)
- `matena2022fisher` (Fisher-weighted, NeurIPS 2022)
- `ortizjimenez2023disentanglement` (NeurIPS 2023 Oral)
- `shannon1959rd` (classical)
- `coverthomas` (textbook)
- `elgamalcover1982md` (classical)
- `yao1977` (classical)

## Lowest priority — placeholder cites in related work only

These can stay as placeholders for arXiv v1 without hurting
the paper — fix in v2 before ICLR submission:
- `jang2025taskvectorbases` (Jang, arXiv:2502.01015)
- `gradients2025taskvectorsgradients` (arXiv:2508.16082)
- `atm2024alternating` (ATM)
- `tseng2024quip` (QuIP#)
- `ashkboos2024quarot` (QuaRot)
- `liu2025spinquant` (SpinQuant)
- `xi2024rolora` (RoLoRA)

## How to fix an entry — worked example

Open `arxiv/v1/references.bib`, find the entry, then:

**Before:**
```bibtex
@article{zandieh2025turboquant,
  title   = {{T}urbo{Q}uant: Randomized Rotation + Optimal Scalar
             Quantization Achieves Near-Shannon-LB Rate Distortion},
  author  = {Zandieh, Amir and Daliri, {Authors TBD} and Hadian,
             {Authors TBD} and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025},
}
```

**After** (hypothetical, replace with what arXiv actually lists):
```bibtex
@article{zandieh2025turboquant,
  title   = {{TurboQuant}: Online Vector Quantization with Hadamard
             Incoherence Processing},
  author  = {Zandieh, Amir and Daliri, Majid and Han, Insu and
             Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025},
}
```

**Also remember to mirror any fix** into `paper/references.bib` (the
working copy) so the two directories stay in sync:

```
cp arxiv/v1/references.bib paper/references.bib
```
