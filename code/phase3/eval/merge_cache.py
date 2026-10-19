"""Cache merged adapters so NLL and downstream cells share one merge.

The NLL cell and the two downstream cells of a (method, cohort) perform the
identical merge. For the expensive methods (TSV-M's full SVDs, MaTS's KFAC
Grams and CG, RegMean's activation Grams) that is the dominant cost, so the
first cell to finish writes the merged factors and the others load them.

The key covers everything the merge depends on: method, kwargs (minus the
Gram tensors themselves, whose inputs are in the kwargs), base model, adapter
directories in order, weights, and the task configs that define the Gram
data. Merges are deterministic given these, so a hit returns the same factors
the cell would have computed. Writes are atomic.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from merging.tests._fake_model import FakeLoraLayer

EXPENSIVE = {"tsv", "mats", "regmean_act", "wudi"}


def cache_key(cfg: dict, method: str, kwargs: dict, weights) -> str:
    blob = {"method": method,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ("grams", "grams_out")},
            "base": cfg["base_model"], "weights": list(weights),
            "adapters": [[s["name"], s["dir"], s.get("task_cfg")] for s in cfg["adapter_specs"]],
            "seed": cfg.get("seed"), "max_seq_length": cfg.get("max_seq_length"),
            "version": 1}
    # Data-driven merges collect Grams through a forward pass, which the two
    # loaders compute with slightly different kernels; keep them apart.
    if method in ("mats", "regmean_act"):
        blob["loader"] = cfg.get("loader")
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:20]


def cache_dir() -> Path:
    return Path(os.environ["PROJECT_ROOT"]) / "results" / "phase3" / "merge_cache"


def load(view, merged_name: str, key: str):
    f = cache_dir() / f"{key}.safetensors"
    if not f.exists():
        return None
    sd = load_file(str(f))
    factors = {layer: FakeLoraLayer(A=sd[f"{layer}::A"], B=sd[f"{layer}::B"], scaling=1.0)
               for layer in view.layer_names()}
    view.add_adapter(merged_name, factors)
    meta = cache_dir() / f"{key}.json"
    return json.loads(meta.read_text()) if meta.exists() else {}


def save(view, merged_name: str, key: str, diagnostics) -> None:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    sd = {}
    for layer, mod in view._layers.items():
        sd[f"{layer}::A"] = mod.lora_A[merged_name].weight.detach().float().cpu().contiguous()
        sd[f"{layer}::B"] = mod.lora_B[merged_name].weight.detach().float().cpu().contiguous()
    tmp = d / f"{key}.tmp{os.getpid()}"
    save_file(sd, str(tmp))
    os.replace(tmp, d / f"{key}.safetensors")
    (d / f"{key}.json").write_text(json.dumps({"diagnostics": diagnostics}, default=str))
