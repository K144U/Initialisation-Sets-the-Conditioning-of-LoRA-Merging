"""Generate E2 multi-seed training configs + orchestrator manifest.

For each of the 16 seed-0 LoRA configs (code/phase3/configs/lora/*.yaml),
emit seed-1 and seed-2 variants (distinct seeds, namespaced output paths,
dataset_num_proc forced to 1 so 5+ concurrent workers don't oversubscribe
the orchestrator job's 8 ncpus). Writes:
  code/phase3/configs/lora_seeds/<name>_seed{1,2}.yaml   (32 configs)
  code/phase3/configs/e2_manifest.json                   (32 cells)

Idempotency: a cell's done-file is its results_path. Seed values are the
literal integers 1 and 2 (seed 0 = the original 20260519 runs), recorded
in each config.

Run: python code/phase3/scripts/gen_e2_manifest.py
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "code/phase3/configs/lora"
DST = ROOT / "code/phase3/configs/lora_seeds"
MANIFEST = ROOT / "code/phase3/configs/e2_manifest.json"
SEEDS = [1, 2]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    cells = []
    base_cfgs = sorted(SRC.glob("*.yaml"))
    print(f"[e2] {len(base_cfgs)} base configs")
    for cfg_path in base_cfgs:
        cfg = yaml.safe_load(cfg_path.read_text())
        stem = cfg_path.stem  # e.g. mistral_7b_alpaca
        for seed in SEEDS:
            c = yaml.safe_load(cfg_path.read_text())  # fresh copy
            c["seeds"] = {"global": seed, "train": seed}
            c["train"]["dataset_num_proc"] = 1
            ad = Path(cfg["output"]["adapter_dir"])
            rp = Path(cfg["output"]["results_path"])
            c["output"]["adapter_dir"] = str(ad.parent / f"seed{seed}")
            c["output"]["results_path"] = str(
                rp.parent / f"{rp.stem}_seed{seed}.json")
            out = DST / f"{stem}_seed{seed}.yaml"
            out.write_text(yaml.safe_dump(c, sort_keys=False))
            cells.append({
                "name": f"e2_{stem}_seed{seed}",
                "cmd": (f"python code/phase3/training/train_lora.py "
                        f"--config {out.relative_to(ROOT)}"),
                "done": c["output"]["results_path"],
                "min_free_gb": float(c["train"].get("min_free_gb", 35)),
            })
    MANIFEST.write_text(json.dumps(cells, indent=1))
    print(f"[e2] wrote {len(cells)} cells -> {MANIFEST}")


if __name__ == "__main__":
    main()
