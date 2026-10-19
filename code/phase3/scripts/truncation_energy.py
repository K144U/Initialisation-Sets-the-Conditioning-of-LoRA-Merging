#!/usr/bin/env python3
"""H1 of notes/prereg_untruncated_2026-10-19.md (CPU).

For the five heuristics, four bases and six cohorts: the fraction of the merged
update's squared norm that rank-16 truncation discards, averaged over the 8
sampled q_proj layers. Merges run on a CPU FakePeftModel built from the adapter
files with the published TOST settings (read from the heur_downstream configs);
the pre-truncation update is captured by wrapping svd_truncate_to_rank in each
method's module.
"""
from __future__ import annotations

import importlib
import json
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merging.registry import DEFAULT_KWARGS, REGISTRY  # noqa: E402
from merging._diag import diag_layers  # noqa: E402
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel  # noqa: E402
from _c1019 import BASES, COHORTS, RES, ROOT, load_cfg  # noqa: E402

TAGS = ["task_arithmetic", "ties", "dare", "tvq_b2", "knots_ties"]
MODULES = {"task_arithmetic": "merging.task_arithmetic", "ties": "merging.ties",
           "dare": "merging.dare", "tvq": "merging.tvq", "knots": "merging.knots"}
TASK_DIRS = ["gsm8k", "alpaca", "magicoder", "flores"]
R = 16


def build(base, cohort):
    sds = [load_file(str(ROOT / "artifacts" / "lora" / base / t / cohort / "adapter_model.safetensors"))
           for t in TASK_DIRS]
    cfgj = json.loads((ROOT / "artifacts" / "lora" / base / "gsm8k" / cohort / "adapter_config.json").read_text())
    scale = cfgj["lora_alpha"] / cfgj["r"]
    names = sorted({k.split(".lora_A.")[0] for k in sds[0] if ".lora_A." in k})
    layers = diag_layers(names)
    m = FakePeftModel()
    for ln in layers:
        A = sds[0][ln + ".lora_A.weight"]
        B = sds[0][ln + ".lora_B.weight"]
        m.layer_topology[ln] = (B.shape[0], A.shape[1], A.shape[0])
    for i, sd in enumerate(sds):
        m.add_adapter(f"t{i}", {ln: FakeLoraLayer(A=sd[ln + ".lora_A.weight"].float(),
                                                   B=sd[ln + ".lora_B.weight"].float(),
                                                   scaling=scale) for ln in layers})
    return m


def main() -> int:
    captured = []
    for mod in MODULES.values():
        M = importlib.import_module(mod)
        orig = M.svd_truncate_to_rank

        def rec(delta, rank, _orig=orig):
            captured.append(delta.detach().double())
            return _orig(delta, rank)
        M.svd_truncate_to_rank = rec
    out = {}
    for base in BASES:
        for cohort in COHORTS:
            m = build(base, cohort)
            names = [f"t{i}" for i in range(4)]
            for tag in TAGS:
                c = load_cfg(RES / "eval_heur_downstream" / f"{base}__{tag}_gsm8k__seed1.json")
                kw = dict(DEFAULT_KWARGS.get(c["method"], {}))
                kw.update(c.get("method_kwargs", {}))
                w = c.get("weights", [0.25] * 4)
                captured.clear()
                REGISTRY[c["method"]](m, names, w, "merged", **kw)
                m.delete_adapter("merged")
                fr = []
                for d in captured:
                    s2 = torch.linalg.svdvals(d) ** 2
                    fr.append(float(1 - s2[:R].sum() / s2.sum()))
                out[f"{base}/{cohort}/{tag}"] = st.mean(fr)
                print(f"{base} {cohort} {tag:16s} discarded {st.mean(fr):.4f}", flush=True)
    (RES / "truncation_energy.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
