"""Shared helpers for the 2026-10-19 red-team generators.

The sanjay.g tree's adapters and base models were emptied on 2026-10-10, so
every template config (embedded in a published result JSON) is re-pointed at
the student account's re-staged copies before a cell is written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
CFG = ROOT / "code" / "phase3" / "configs"
OLD = "/home/sanjay.g/projects/rdmerge"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SHARED = ["seed1", "seed2", "seed3"]
INDEP = ["indep1", "indep2", "indep3"]
COHORTS = SHARED + INDEP
INVARIANT = ["max_seq_length", "seed", "loader"]
# nll_tau cache for re-staged models; seeded from nll_tau_cache_v2 only if
# the restage check (gen_restage_check.py) matched the published values.
CACHE = RES / os.environ.get("RDMERGE_TAU_CACHE", "nll_tau_cache_v2")


def restage(s: str) -> str:
    return (s.replace(f"{OLD}/models/", f"{ROOT}/models/")
             .replace(f"{OLD}/artifacts/lora/", f"{ROOT}/artifacts/lora/"))


def load_cfg(p: Path) -> dict:
    return json.loads(p.read_text())["meta"]["config"]


def template(p: Path) -> dict:
    cfg = load_cfg(p)
    cfg["base_model"] = restage(cfg["base_model"])
    cfg["adapter_specs"] = [dict(s, dir=restage(s["dir"])) for s in cfg["adapter_specs"]]
    if not Path(cfg["base_model"], "config.json").exists():
        raise SystemExit(f"base model not staged: {cfg['base_model']}")
    return cfg


def retarget(specs: list[dict], src: str, dst: str, check: bool = True) -> list[dict]:
    out = []
    for s in specs:
        s = dict(s)
        d = s["dir"].rstrip("/")
        if not d.endswith("/" + src):
            raise SystemExit(f"template dir {d} does not end in {src}")
        s["dir"] = d[: -len(src)] + dst
        if check and not Path(s["dir"], "adapter_model.safetensors").exists():
            raise SystemExit(f"missing adapter {s['dir']}")
        out.append(s)
    return out


class Manifest:
    def __init__(self):
        self.cells = []

    def add(self, name, cfg, cfg_dir, runner, min_free=30.0):
        (CFG / cfg_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        p = CFG / cfg_dir / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.cells.append({
            "name": name,
            "cmd": f"python code/phase3/eval/{runner} --config {p.relative_to(ROOT)}",
            "done": str(Path(cfg["output_path"]).relative_to(ROOT)),
            "min_free_gb": min_free,
        })

    def write(self, fname, expect=None):
        if expect is not None and len(self.cells) != expect:
            raise SystemExit(f"{fname}: expected {expect} cells, built {len(self.cells)}")
        out = CFG / fname
        out.write_text(json.dumps(self.cells, indent=1))
        print(f"wrote {out}: {len(self.cells)} cells")
        return out
