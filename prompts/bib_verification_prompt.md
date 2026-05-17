# Prompt for Claude Opus (web, with Search/Fetch tools)

*Paste everything below the `--- PROMPT START ---` line into a fresh
Claude Opus conversation on claude.ai. Opus has web search and
URL-fetch tools; this prompt instructs it to use them to canonicalize
16 broken BibTeX entries.*

*When Opus is done, copy the BibTeX block from its response and
replace the corresponding entries in `arxiv/v1/references.bib` and
`paper/references.bib`.*

---

--- PROMPT START ---

I'm finalizing a research paper for arXiv submission and I need you
to verify and canonicalize 16 BibTeX entries. The paper is on
rate-distortion theory for LoRA model merging; the bibliography
currently has placeholder author names ("Authors TBD") that I need
replaced with the real author lists from the canonical sources
(arXiv abstract pages, OpenReview, NeurIPS / ICLR / ICML / ICCV
proceedings, or DBLP).

Please use your web search and fetch tools to look each one up.
Source priority:
1. **arXiv abstract page** (`arxiv.org/abs/<id>`) when an arXiv ID
   is given — this gives the canonical title and author list.
2. **Venue proceedings** (OpenReview / NeurIPS / ICLR / ICML / ICCV
   open-access) when the paper is published — use the venue's BibTeX
   export if available, or DBLP.
3. **Google Scholar / Semantic Scholar** as a fallback search tool
   only — never copy a citation directly from there without
   verifying the title and authors against the primary source.

**Honesty rules:**
- Do NOT invent authors. If a paper is anonymous-under-review on
  OpenReview (no public author list), say so and return the entry
  with `note = {Anonymous OpenReview submission; could not verify
  authors.}`.
- If you cannot find a paper at all (e.g., the title may be
  misremembered), return the original entry with `note = {Could
  not locate; consider dropping or finding a closer match.}` and
  flag it explicitly in your prose summary.
- If the paper exists but has been retitled between arXiv versions,
  use the title from the latest version and note the version in
  the comment above the entry.
- Cross-check the year: arXiv submission year, not preprint year.

## Output format

Return a single fenced ```bibtex``` code block containing only the
16 corrected entries (no other entries — I'll merge them into my
existing `references.bib` myself). For each entry:

- Preserve the citation key exactly (e.g. `zandieh2025turboquant`
  stays as is — do not rename).
- Choose the appropriate entry type: `@inproceedings` if published
  in a venue, `@article` if arXiv-only or in a journal.
- Required fields: `title`, `author`, then either `booktitle` (for
  conferences) or `journal` (for arXiv / journals), `year`. Add
  `pages`, `volume`, `number` if available from the venue
  publication.
- Add `eprint = {<arXiv ID>}, archivePrefix = {arXiv}` for arXiv
  papers — even ones that ended up in proceedings, if the arXiv
  version is the more accessible one.
- Add a one-line `% comment` ABOVE each entry summarizing the
  source you used (e.g. `% Verified from arXiv:2504.19874 abs page,
  fetched 2026-MM-DD`).

After the BibTeX block, write a brief prose summary (~1 paragraph)
listing any entries you couldn't fully verify and why.

## The 16 entries to verify

For each: I'm giving you the citation key, the current placeholder
state, and a search starting point. Verify and replace.

### High priority (cited multiple times in the paper body)

**1. `zandieh2025turboquant`**
- arXiv ID: `2504.19874`
- Current placeholder authors: `Zandieh, Amir and Daliri, {Authors
  TBD} and Hadian, {Authors TBD} and Mirrokni, Vahab`
- Verify: full author list (likely 4–5 authors), exact canonical
  title (the abstract may use a different phrasing than what's in
  the placeholder).

**2. `systematic2025merging`**
- arXiv ID: `2511.21437`
- Current placeholder authors: `{Authors TBD}`
- Verify: authors, exact title, year (Nov 2025).

**3. `stoica2025knots`**
- Venue: ICLR 2025
- Current authors (likely incomplete): `Stoica, George and Ramesh,
  Pratik and Shah, Viraj`
- Verify: full author list on OpenReview; title is "KnOTS: Aligning
  LoRA Adapters via SVD for Effective Model Merging" — confirm this
  is correct.

**4. `panariello2025core`**
- Title hint: "Core Space" + LoRA merging + Stiefel
- Current authors: `Panariello, Antonio and others`
- Search arXiv and OpenReview; verify Antonio Panariello is the
  first author, get the rest.

### Medium priority (one-off citations in related work)

**5. `tspa2025`** — "Leveraging Rotation Symmetry for Efficient
LoRA Merging (TSPA)". Search OpenReview 2025 cycle. May be
anonymous — if so, return as such.

**6. `domerging2025`** — "DO-Merging: Decoupled Orthogonal
Perturbation Merging of LoRA Adapters", NeurIPS 2025. Search
NeurIPS 2025 proceedings.

**7. `arm2026streaming`**
- Title hint: "Activation-Guided Rotation for Streaming LoRA
  Merging (ARM)"
- arXiv ID I have is `2602.03237` — but `2602` is a suspicious
  prefix (arXiv IDs usually start `2YMM` where YY = year). Please
  verify the arXiv ID first; it may be a typo for `2502.03237` or
  similar.

**8. `tara2025`** — "TARA-Merging: Directional Anisotropy in LoRA
Merging". Search arXiv.

**9. `kim2025tvq`**
- Venue: ICCV 2025
- Title: "Task-Vector Quantization for Memory-Efficient Model
  Merging (TVQ)"
- Verify: "Kim" is the first author; get full list and exact
  ICCV 2025 cite.

**10. `bit1merging2025`** — "1bit-Merging: Module-wise One-Bit
Quantization for Task-Vector Merging with Routing". Search arXiv.

### Lower priority (related-work mentions, less critical)

**11. `jang2025taskvectorbases`**
- arXiv ID: `2502.01015`
- Current authors: `Jang, {Authors TBD}`
- Verify: "Jang" is first author; get full list.

**12. `gradients2025taskvectorsgradients`**
- arXiv ID: `2508.16082`
- Title: "On Task Vectors and Gradients"
- Current authors: `{Authors TBD}`
- Get full author list.

**13. `atm2024alternating`** — "ATM: Alternating Tuning and Merging
for Multi-Task Adaptation". Search arXiv 2024.

**14. `tseng2024quip`** — "QuIP\#: Even Better LLM Quantization
with Hadamard Incoherence and Lattice Codebooks". Verify Tseng is
first author; this paper is well-known in LLM quantization, should
be easy to find.

**15. `ashkboos2024quarot`**
- Title: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
- Current authors: `Ashkboos, Saleh and {Authors TBD}`
- Verify and complete author list.

**16. `liu2025spinquant`** and **`xi2024rolora`**
- SpinQuant (Liu first author, 2025) and RoLoRA (Xi first author,
  2024). Both are LLM-quantization papers. Get full cites.

## Entries you should NOT touch

For context, these 11 entries in my bibliography are already
correctly cited; do not rewrite them:

- `hu2022lora` (LoRA, ICLR 2022)
- `ilharco2023task` (Task Arithmetic, ICLR 2023)
- `yadav2023ties` (TIES, NeurIPS 2023)
- `yu2024dare` (DARE, ICML 2024)
- `matena2022fisher` (Fisher-weighted, NeurIPS 2022)
- `ortizjimenez2023disentanglement` (NeurIPS 2023 Oral)
- `shannon1959rd` (Shannon 1959)
- `coverthomas` (Cover & Thomas textbook, 2nd ed.)
- `elgamalcover1982md` (El Gamal & Cover 1982 IEEE-IT)
- `yao1977` (Yao FOCS 1977)

Don't return entries for these; only return the 16 above.

## Worked example (input → output)

**Input** (broken):
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

**Expected output** (after fetching `arxiv.org/abs/2504.19874`):
```bibtex
% Verified from arXiv:2504.19874 abs page, fetched 2026-04-25
@article{zandieh2025turboquant,
  title         = {<canonical title from arXiv>},
  author        = {<full author list as comma-separated, "Last, First"
                   pairs joined by " and ">},
  journal       = {arXiv preprint arXiv:2504.19874},
  year          = {2025},
  eprint        = {2504.19874},
  archivePrefix = {arXiv},
}
```

(I deliberately left the title and authors as `<...>` placeholders
in this example — the real output should have the actual values
you fetch from arXiv.)

## Begin

Please proceed: fetch each source, build the BibTeX block, and
return it followed by a one-paragraph prose summary of any entries
you couldn't fully verify.

--- PROMPT END ---

## How to use the output

When Opus finishes:

1. Copy the BibTeX block from its response.
2. Open `arxiv/v1/references.bib`.
3. For each entry in the block, find the matching `@article{key,...}`
   or `@inproceedings{key,...}` block in the file and replace it
   wholesale with Opus's version.
4. Mirror the changes:
   ```
   cp arxiv/v1/references.bib paper/references.bib
   ```
5. Recompile on Overleaf — references on pages 14–16 should now
   show real author names instead of "Authors TBD".

If Opus flags anything as unverifiable in its prose summary, you
have three choices for that entry:
- **Drop it from the body**: open the relevant `paper/sections/*.tex`
  file, remove the `\cite{key}` reference, and let the entry stay
  in `.bib` (BibTeX silently ignores unused entries).
- **Keep it as `Authors TBD`**: better than fabricated authors;
  reviewers will know it's a placeholder.
- **Replace with a different paper**: if the original placeholder
  was a misidentification, find the actual paper that argues the
  point you wanted and cite that instead.
