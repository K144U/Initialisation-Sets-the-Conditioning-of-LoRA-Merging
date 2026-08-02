#!/usr/bin/env python3
"""W5 / audit A4+A5: re-run the downstream matrix with the fixed scorers.

Two harness bugs made the published downstream table a measure of output
formatting rather than task ability (notes/audit_2026-08-02_code_and_claims.md):

  A5  gsm8k_extract_answer required the final number at end-of-string, so any
      answer closing with a word scored 0. Failure rates on the shipped runs
      were 60% to 81% on three of four bases and method-dependent (Llama
      rd-ridge 0.693 vs Mistral rd-ridge 0.107). This is the most likely
      cause of the review's W5, the (Llama-3.1, GSM8K EM) outlier that the
      paper lists as unexplained.

  A4  _strip_humaneval_completion returned "" for any markdown-fenced answer,
      discarding 122 to 130 of 164 completions for TA/DARE/KnOTS versus 0 to
      11 for TIES/TVQ2/rd-ridge.

Both are fixed in eval/downstream_metrics.py (22 self-test cases). The scores
cannot be recomputed offline because only 200-char previews were stored, so
the cells must re-run. They now store FULL generations, so any future scorer
change is a CPU re-score rather than another 36 GPU-hours.

Cells are reconstructed from the *published result JSONs*, each of which
embeds its own originating config under meta.config, rather than from the
config directories. That guarantees the re-run is configuration-identical to
whatever produced the published number, and it does not depend on any config
directory still being present or correctly named.

  eval_e3_gsm8k_seed            60   5 methods x 4 bases x 3 seeds
  eval_b4_humaneval_seed        60
  eval_e3b_gsm8k_rdridge_seed   12   rd-ridge, 4 bases x 3 seeds
  eval_b4b_humaneval_rdridge_seed 12
                               ---
                               144 cells, ~10-25 min each

Output goes to results/phase3/eval_downstream_v2/<original name>.json. The
published results are left untouched so the before/after delta is auditable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

# Published downstream result directories; each JSON carries meta.config.
SOURCE_RESULT_DIRS = [
    "eval_e3_gsm8k_seed",
    "eval_b4_humaneval_seed",
    "eval_e3b_gsm8k_rdridge_seed",
    "eval_b4b_humaneval_rdridge_seed",
]
EXPECTED = {"eval_e3_gsm8k_seed": 60, "eval_b4_humaneval_seed": 60,
            "eval_e3b_gsm8k_rdridge_seed": 12,
            "eval_b4b_humaneval_rdridge_seed": 12}

OUT_CFG = CFG / "eval_downstream_v2"
OUT_RES = RES / "eval_downstream_v2"


def main() -> int:
    manifest: list[dict] = []
    problems: list[str] = []
    counts: dict[str, int] = {}

    for src in SOURCE_RESULT_DIRS:
        src_dir = RES / src
        if not src_dir.is_dir():
            problems.append(f"missing result dir {src_dir}")
            continue
        results = sorted(src_dir.glob("*.json"))
        counts[src] = len(results)
        if len(results) != EXPECTED.get(src):
            problems.append(f"{src}: found {len(results)} results, "
                            f"expected {EXPECTED.get(src)}")
        for res_p in results:
            payload = json.loads(res_p.read_text())
            cfg = payload.get("meta", {}).get("config")
            if not cfg:
                problems.append(f"no meta.config in {res_p.name}")
                continue
            for spec in cfg.get("adapter_specs", []):
                if not Path(spec["dir"]).exists():
                    problems.append(f"missing adapter {spec['dir']}")
            name = res_p.stem
            out_json = OUT_RES / f"{name}.json"
            cfg["output_path"] = out_json.as_posix()
            new_p = OUT_CFG / f"{name}.yaml"
            new_p.parent.mkdir(parents=True, exist_ok=True)
            new_p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            manifest.append({
                "name": f"w5_{name}",
                "cmd": f"python code/phase3/eval/run_downstream_cell.py "
                       f"--config {new_p.relative_to(ROOT).as_posix()}",
                "done": out_json.relative_to(ROOT).as_posix(),
                "min_free_gb": 25.0,
            })

    man_p = CFG / "w5_rescore_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2))
    print(f"[gen] {len(manifest)} cells -> {man_p}")
    for k, v in sorted(counts.items()):
        print(f"[gen]   {k:<34}{v:>4}")
    if problems:
        print("[gen] PROBLEMS:")
        for p in sorted(set(problems))[:12]:
            print("  - " + p)
        extra = len(set(problems)) - 12
        if extra > 0:
            print(f"  ... and {extra} more")
        return 1
    print("[gen] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
