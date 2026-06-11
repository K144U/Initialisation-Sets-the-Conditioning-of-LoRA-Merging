"""Generate E1 eval-cell configs + orchestrator manifest (encoder on real adapters).

Cells: 4 models x bits in {1, 2, 3, 4, 8, 16, 32}  (32 = b-infinity: the
unquantized H-weighted centroid, the experiment's most diagnostic point).
H-variant: projector surrogate (the deff_analysis geometry). The diagonal-
Fisher variant is a second pass once projector results are in.

Reuses the v3 per-model eval configs as templates (same tasks, splits,
n_eval=1000, seed), swapping method/method_kwargs/output_path. Writes
  code/phase3/configs/eval_e1/<model>__rd_b<bits>.yaml      (28 configs)
  code/phase3/configs/e1_manifest.json                      (28 cells)
  code/phase3/configs/all_manifest.json                     (E2 + E1, for
                                                             the keeper)
Run: python code/phase3/scripts/gen_e1_configs.py
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
TPL_DIR = ROOT / "code/phase3/configs/eval_n1k_v3_perexample"
DST = ROOT / "code/phase3/configs/eval_e1"
E1_MANIFEST = ROOT / "code/phase3/configs/e1_manifest.json"
E2_MANIFEST = ROOT / "code/phase3/configs/e2_manifest.json"
ALL_MANIFEST = ROOT / "code/phase3/configs/all_manifest.json"
OUT_DIR = "results/phase3/eval_e1"

MODELS = ["llama31_8b", "qwen25_7b", "mistral_7b", "yi15_9b"]
BITS = [1, 2, 3, 4, 8, 16, 32]
SEED = 20260611


def main():
    DST.mkdir(parents=True, exist_ok=True)
    (ROOT / OUT_DIR).mkdir(parents=True, exist_ok=True)
    cells = []
    for model in MODELS:
        tpl_path = TPL_DIR / f"{model}__task_arithmetic.yaml"
        if not tpl_path.exists():
            cands = sorted(TPL_DIR.glob(f"{model}__*.yaml"))
            assert cands, f"no template for {model}"
            tpl_path = cands[0]
        for b in BITS:
            cfg = yaml.safe_load(tpl_path.read_text())
            cfg["method"] = "rd_encoder"
            cfg["method_kwargs"] = {"bits": b, "c": 5.0, "seed": SEED}
            out = f"{OUT_DIR}/{model}__rd_b{b}.json"
            cfg["output_path"] = str(ROOT / out)
            cfg_path = DST / f"{model}__rd_b{b}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            cells.append({
                "name": f"e1_{model}_rd_b{b}",
                "cmd": (f"python code/phase3/eval/run_eval_cell.py "
                        f"--config {cfg_path.relative_to(ROOT)}"),
                "done": out,
                "min_free_gb": float(cfg.get("min_free_gb", 25)),
            })
    E1_MANIFEST.write_text(json.dumps(cells, indent=1))
    e2 = json.loads(E2_MANIFEST.read_text()) if E2_MANIFEST.exists() else []
    ALL_MANIFEST.write_text(json.dumps(e2 + cells, indent=1))
    print(f"[e1] {len(cells)} cells -> {E1_MANIFEST}")
    print(f"[all] {len(e2)} E2 + {len(cells)} E1 -> {ALL_MANIFEST}")


if __name__ == "__main__":
    main()
