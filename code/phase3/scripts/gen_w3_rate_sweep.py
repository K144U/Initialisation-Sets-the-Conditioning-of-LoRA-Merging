#!/usr/bin/env python3
"""W3: the finite-rate sweep of rd-encoder ridge on real adapters.

The review's objection: RD-ENCODER RIDGE is only ever evaluated at b -> inf,
where the floor-zero lower bound is identically zero, so the rate-distortion
machinery contributes nothing quantitative to the headline. The only real-data
rate sweep in the paper (TVQ) is non-monotone on all four bases.

Worse than the reviewer knows: the encoder's OWN rate sweep already exists at
results/phase3/eval_e1/ and was never shown as a rate curve. It is non-monotone
on 4 of 4 bases and broken outright on Mistral (9-11 nats at every b):

    base       b=1     b=2     b=3     b=4     b=8    b=16    b=32
    Llama   11.417   0.925   0.356   0.405   0.505   0.468   0.497
    Mistral 11.672   9.472   9.917   9.135   9.106   9.771  10.778
    Qwen     9.261   1.728   0.349   0.232   0.210   0.168   0.191
    Yi      13.537   5.420   0.524   5.035   0.613   0.251   0.376

But those cells are lambda = 0, i.e. the unregularised construction that the
paper itself shows collapses, so they conflate the sliver blow-up with the rate
axis. The experiment the review actually asks for, ridge ON at finite b, has
never been run: every rd-ridge cell in the paper is bits=32.

This generates it: b in {1, 2, 3, 4, 8, 16} at each base's lambda*, seed1.
b = 32 (the b -> inf reference) already exists in the published cells.

  6 rates x 4 bases = 24 cells, ~12-18 min each.

realize="rank_deff" throughout, deliberately. Truncating to rank 16 adds a
rate-INDEPENDENT error floor that would flatten the curve and destroy the very
slope being measured; rank_deff carries W* exactly, so the only distortion is
quantization. This is a rate experiment, not a storage-parity experiment
(that one is W1 group C).

THE PREDICTION, fixed before the cells run. The encoder quantizes eta, which
is (out x d_eff), at b bits per entry, so R = b * out * d_eff over n =
out * d_eff dimensions and the theory's 2^{-2R/n} becomes 2^{-2b}: excess
should fall 4x per added bit, i.e. slope -2 in log2(excess) vs b. That is the
same prediction the synthetic validation confirms at -2.00 +/- 0.01
(App. A). The quantity to fit is the QUANTIZATION contribution,
excess(b) - excess(inf), because excess(inf) is the merge error itself and
does not vanish with rate.

  slope in [-2.4, -1.6]  -> the rate axis behaves as the theory says on real
                            adapters, and the paper gains its first real-data
                            confirmation of the achievability exponent
  slope flat or positive -> the quadratic surrogate does not describe real
                            merging in the rate regime where the theory has
                            content, and Theorem 4's empirical status must be
                            restated in the abstract, not just conceded in 6.1
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BITS = [1, 2, 3, 4, 8, 16]          # b=32 reference already published
SEED = "seed1"
LAMBDA_STAR = {"llama31_8b": 0.05, "mistral_7b": 0.13,
               "qwen25_7b": 0.13, "yi15_9b": 0.13}

OUT_CFG = CFG / "eval_w3_rate"
OUT_RES = RES / "eval_w3_rate"


def main() -> int:
    manifest: list[dict] = []
    problems: list[str] = []

    for base in BASES:
        tmpl_p = CFG / f"eval_matrix_seeds/{base}__ties__{SEED}.yaml"
        if not tmpl_p.exists():
            problems.append(f"missing template {tmpl_p}")
            continue
        tmpl = yaml.safe_load(tmpl_p.read_text())
        for spec in tmpl["adapter_specs"]:
            if not Path(spec["dir"]).exists():
                problems.append(f"missing adapter {spec['dir']}")
        for b in BITS:
            cfg = dict(tmpl)
            cfg["method"] = "rd_encoder"
            cfg["method_kwargs"] = {"bits": b, "c": 5.0, "seed": 20260611,
                                    "realize": "rank_deff",
                                    "ridge_lambda": LAMBDA_STAR[base]}
            cfg["loader"] = "plain"          # rank_deff is a mixed-rank adapter
            cfg.pop("weights", None)
            name = f"{base}__rd_ridge_b{b}__{SEED}"
            out_json = OUT_RES / f"{name}.json"
            cfg["output_path"] = out_json.as_posix()
            p = OUT_CFG / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            manifest.append({
                "name": f"w3_{name}",
                "cmd": f"python code/phase3/eval/run_eval_cell.py "
                       f"--config {p.relative_to(ROOT).as_posix()}",
                "done": out_json.relative_to(ROOT).as_posix(),
                "min_free_gb": 25.0,
            })

    man_p = CFG / "w3_rate_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2))
    print(f"[gen] {len(manifest)} cells -> {man_p}")
    print(f"[gen] bits {BITS} at lambda* {LAMBDA_STAR}, realize=rank_deff")
    if problems:
        print("[gen] PROBLEMS:")
        for p in sorted(set(problems))[:10]:
            print("  - " + p)
        return 1
    print("[gen] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
