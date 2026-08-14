"""Seed the nll_tau cache from a cell that already ran.

nll_tau is merge-independent and deterministic (check_nll_tau_determinism.py:
960 comparisons, worst difference exactly 0), so a completed cell's nll_tau is
exactly what a fresh run under the same config would produce. This writes it
into the cache so the next cell of that cohort does not repeat 20 evaluations.

The key is rebuilt from the CONFIG, not from the result, and must match what
run_eval_cell.py computes. If the two ever drift, the cache misses and the cell
recomputes, which is the safe direction.

The third argument exists because a cell can be a valid source for a cache
entry its own config knows nothing about. The eval_a1_indep baseline cells ran
on the unsloth path long before this cache existed, and their nll_tau is
exactly what an unsloth-path cell of the same cohort needs.

Usage:
  python code/phase3/scripts/seed_nll_tau_cache.py <config.yaml> <result.json>
                                                   [cache_path]
"""
import json
import sys
from pathlib import Path

import yaml


def main(cfg_path, res_path, override=None):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    res = json.loads(Path(res_path).read_text())

    cache_path = override or cfg.get("nll_tau_cache")
    if not cache_path:
        sys.exit(f"{cfg_path} has no nll_tau_cache field and none was given")
    cache_path = Path(cache_path)

    n_eval = res.get("n_eval_per_task")
    if not n_eval:
        sys.exit("result has no n_eval_per_task; cannot rebuild the key")

    key = {
        "base_model": cfg["base_model"],
        "adapters": [[s["name"], s["dir"]] for s in cfg["adapter_specs"]],
        "task_cfgs": [s["task_cfg"] for s in cfg["adapter_specs"]],
        "seed": cfg.get("seed", 20260518),
        "max_seq_length": cfg.get("max_seq_length", 1024),
        "n_eval": n_eval,
        "loader": cfg.get("loader"),
        "delta_scale": cfg.get("delta_scale"),
        "version": 1,
    }

    # Sanity: the result must come from the same adapters as the config.
    if [s["dir"] for s in cfg["adapter_specs"]] != res["adapter_dirs"]:
        sys.exit("config and result disagree on adapter_dirs; refusing to seed")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "key": key,
        "nll_tau": res["nll_tau"],
        "per_example_tau": res["per_example_nll_tau"],
    }))
    print(f"seeded {cache_path} from {Path(res_path).name}")
    print(f"  states {list(res['nll_tau'])}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
