# Final two verdicts, 2026-09-28 08:39 IST (not yet written into the paper)

- regmean_real: branch 1. Q4 holds (lambda 0.01 beats shipped alpha 0.9 in
  accuracy on shared arm: GSM8K +0.31/+0.43/+0.54 Llama/Qwen/Yi; HumanEval
  Llama, Qwen). Q5 holds (real RegMean no accuracy effect, 16/16 cells).
  NOTE: shipped alpha 0.9 scores 0.000 on Mistral GSM8K and HumanEval on all
  three shared cohorts, but lambda 0.01 does NOT rescue Mistral either
  (0.018-0.040 GSM8K); Mistral's selected ridge is 0.05 (Table 4). Say so.
- heuristics_tost: E1, E2 hold (all 5 methods equivalent within 3 pts GSM8K,
  5 pts HumanEval; smallest margins 0.009-0.030), E3 holds (1 effect cell),
  E4 FAILS: 29 of 38 nonzero differences favour the SHARED arm, p = 0.002.
  Branch 4: report the lean as a registered finding in its direction.
- Remaining: write both into exposure.tex (subsec:ridge, regmean-default)
  and intro/abstract; trim abstract 262 -> <=250 words; sync results; rebuild
  public repo; add new commit hashes to app_prereg table.
