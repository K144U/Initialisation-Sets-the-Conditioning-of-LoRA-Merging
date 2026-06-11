"""Generate the follow-up queues after E2 trainings + E1 v1:

1. E1v2 full-rank cells (decision rule of 2026-06-12: trunc_mass ~0.30 >> 0.1
   threshold, so the b=infinity verdict needs the residual base-patch path):
   4 models x bits in {2, 4, 32} with full_rank_patch=true -> 12 cells,
   configs/eval_e1_fullrank/, results/phase3/eval_e1_fullrank/.

2. E2 merge matrices on the seed-1/2 adapters (master plan: "re-run the full
   merge matrix per seed"): for every v3 template cell (4 models x
   {task_arithmetic, ties, dare, knots, tvq_b1..b32}), two variants with all
   adapter dirs .../v1 -> .../seed{1,2} -> 80 cells,
   configs/eval_matrix_seeds/, results/phase3/eval_matrix_seeds/.

Appends everything to configs/all_manifest.json (idempotent by done-file).
Run: python code/phase3/scripts/gen_followup_cells.py
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
TPL_DIR = ROOT / "code/phase3/configs/eval_n1k_v3_perexample"
ALL_MANIFEST = ROOT / "code/phase3/configs/all_manifest.json"
MODELS = ["llama31_8b", "qwen25_7b", "mistral_7b", "yi15_9b"]
SEED = 20260611


def e1_fullrank_cells():
    dst = ROOT / "code/phase3/configs/eval_e1_fullrank"
    out_dir = "results/phase3/eval_e1_fullrank"
    dst.mkdir(parents=True, exist_ok=True)
    (ROOT / out_dir).mkdir(parents=True, exist_ok=True)
    cells = []
    for model in MODELS:
        tpl = TPL_DIR / f"{model}__task_arithmetic.yaml"
        for b in [2, 4, 32]:
            cfg = yaml.safe_load(tpl.read_text())
            cfg["method"] = "rd_encoder"
            cfg["method_kwargs"] = {"bits": b, "c": 5.0, "seed": SEED,
                                    "full_rank_patch": True}
            out = f"{out_dir}/{model}__rdfr_b{b}.json"
            cfg["output_path"] = str(ROOT / out)
            p = dst / f"{model}__rdfr_b{b}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            cells.append({
                "name": f"e1fr_{model}_rd_b{b}",
                "cmd": (f"python code/phase3/eval/run_eval_cell.py "
                        f"--config {p.relative_to(ROOT)}"),
                "done": out,
                "min_free_gb": float(cfg.get("min_free_gb", 25)),
            })
    return cells


def e2_matrix_cells():
    dst = ROOT / "code/phase3/configs/eval_matrix_seeds"
    out_dir = "results/phase3/eval_matrix_seeds"
    dst.mkdir(parents=True, exist_ok=True)
    (ROOT / out_dir).mkdir(parents=True, exist_ok=True)
    cells = []
    for tpl in sorted(TPL_DIR.glob("*.yaml")):
        base_cfg = yaml.safe_load(tpl.read_text())
        stem = tpl.stem                      # e.g. llama31_8b__tvq_b2
        for s in (1, 2):
            cfg = yaml.safe_load(tpl.read_text())
            for spec in cfg["adapter_specs"]:
                assert spec["dir"].endswith("/v1"), spec["dir"]
                spec["dir"] = spec["dir"][:-3] + f"/seed{s}"
            out = f"{out_dir}/{stem}__seed{s}.json"
            cfg["output_path"] = str(ROOT / out)
            p = dst / f"{stem}__seed{s}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            cells.append({
                "name": f"e2m_{stem}_seed{s}",
                "cmd": (f"python code/phase3/eval/run_eval_cell.py "
                        f"--config {p.relative_to(ROOT)}"),
                "done": out,
                "min_free_gb": float(cfg.get("min_free_gb", 25)),
            })
    return cells


def main():
    fr = e1_fullrank_cells()
    em = e2_matrix_cells()
    manifest = json.loads(ALL_MANIFEST.read_text())
    have = {c["name"] for c in manifest}
    new = [c for c in fr + em if c["name"] not in have]
    manifest.extend(new)
    ALL_MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"[followup] e1-fullrank {len(fr)} cells, e2-matrix {len(em)} cells; "
          f"{len(new)} appended -> all_manifest now {len(manifest)} cells")


if __name__ == "__main__":
    main()
