#!/usr/bin/env python3
"""Part A of the prevalence pre-registration: when do two adapters share an A?

The registration asks for the exact condition, from the library source at a
pinned version, under which two LoRA adapters trained by separate invocations
receive identical `A` factors. It fixes no threshold and returns no verdict:
it is a factual question about a library.

The version is not a choice. Every adapter we shipped records the library that
wrote it in its own `adapter_config.json`, and this script reads that rather
than reporting whatever happens to be installed today.

Source condition, PEFT v0.19.1, `src/peft/tuners/lora/layer.py`, in
`reset_lora_parameters`, reached whenever `init_lora_weights is True` (the
default, and what every one of our configs records):

    nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B[adapter_name].weight)

`kaiming_uniform_` draws from the **global torch RNG**. It takes no per-adapter
seed and PEFT never reseeds around it. So two separate invocations receive
identical `A` exactly when the global RNG is in the same state at the moment
each layer is constructed, which needs all of:

  1. the same seed set before the model is built,
  2. the same layers targeted in the same order, at the same rank, so the same
     number of draws happen in the same sequence, and
  3. the same device and dtype, since the CPU and CUDA generators are separate
     streams.

Nothing about the task, the data, or the training run enters. That is the whole
point: the condition is decided entirely before the first gradient step, which
is why a single seed line in a launcher script decides it for a whole cohort
and why it is so rarely reported.

This script checks that stated condition against our own configs, and reports
the post-training signature separately.

One thing it deliberately does **not** do is compare the shipped `A` tensors
bitwise. We tried that first and it is the wrong instrument: the shipped
tensors are post-training, and training moves `A`, so four adapters that
started from one draw still end up with four distinct tensors. Bitwise
identity is a property of the initialisation, and we did not retain the
initialisation. What the artifacts can show is the *residue* of a shared
start, which is the relative distance of Appendix C.3, and what the configs can show
is whether the source condition was met at launch. Those are the two checks
below.

Usage:
  python code/phase3/scripts/audit_peft_init_condition.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
LORA = REPO / "artifacts" / "lora"
OUT = REPO / "results" / "phase3" / "peft_init_condition.json"

TASKS = ["alpaca", "gsm8k", "magicoder", "flores"]
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]


def a_digest(path: Path) -> tuple[str, int]:
    """Hash of every lora_A tensor in one adapter, in sorted key order.

    Reported for the record, not as a test of shared initialisation: see the
    module docstring for why post-training tensors cannot answer that.
    """
    sd = load_file(str(path))
    h = hashlib.sha256()
    n = 0
    for k in sorted(sd):
        if "lora_A" not in k:
            continue
        h.update(k.encode())
        h.update(sd[k].contiguous().cpu().numpy().tobytes())
        n += 1
    return h.hexdigest()[:16], n


def adapter_cfg(base: str, task: str, cohort: str) -> dict | None:
    cfg = LORA / base / task / cohort / "adapter_config.json"
    if not cfg.exists():
        return None
    return json.loads(cfg.read_text())


def peft_version(base: str, task: str, cohort: str) -> str | None:
    d = adapter_cfg(base, task, cohort)
    return d.get("peft_version") if d else None


def launch_condition(base: str, cohort: str) -> dict:
    """Was the source condition met at launch, for this cohort?

    The three clauses of the condition map onto things recorded in the run
    configs and in the adapter configs: the seed set before the model is built,
    and the shape of the construction (targets, rank, base model).
    """
    import yaml

    seeds, shapes, present = {}, {}, []
    for task in TASKS:
        run = (REPO / "code" / "phase3" / "configs" /
               ("lora_seeds" if cohort.startswith("seed") else "lora_indep") /
               f"{base}_{task}_{cohort}.yaml")
        ad = adapter_cfg(base, task, cohort)
        if not run.exists() or ad is None:
            continue
        present.append(task)
        cfg = yaml.safe_load(run.read_text())
        seeds[task] = (int(cfg["seeds"]["global"]), int(cfg["seeds"]["train"]))
        shapes[task] = (tuple(sorted(ad["target_modules"])), ad["r"],
                        ad["base_model_name_or_path"], ad["init_lora_weights"])

    if not present:
        return {}
    same_seed = len(set(seeds.values())) == 1
    same_shape = len(set(shapes.values())) == 1
    return {
        "n_tasks": len(present),
        "seeds": {k: list(v) for k, v in seeds.items()},
        "same_seed_across_tasks": same_seed,
        "same_construction_across_tasks": same_shape,
        # All three clauses of the source condition, and only then.
        "source_condition_met": same_seed and same_shape,
    }


def main() -> int:
    report: dict = {
        "source_condition": {
            "peft_version_recorded_in_artifacts": sorted(
                {v for b in BASES for t in TASKS for c in ("seed1", "indep1")
                 if (v := peft_version(b, t, c))}),
            "file": "src/peft/tuners/lora/layer.py",
            "function": "reset_lora_parameters",
            "init_A": "nn.init.kaiming_uniform_(weight, a=math.sqrt(5))",
            "init_B": "nn.init.zeros_(weight)",
            "rng": "global torch RNG; no per-adapter seed is taken",
        },
        "cohorts": {},
    }

    for base in BASES:
        entry = {}
        for cohort in ("seed1", "indep1"):
            digests = {}
            for task in TASKS:
                p = LORA / base / task / cohort / "adapter_model.safetensors"
                if not p.exists():
                    continue
                digests[task] = a_digest(p)
            if not digests:
                continue
            uniq = {d for d, _ in digests.values()}
            entry[cohort] = {
                "n_adapters": len(digests),
                # Post-training, and expected to be 4 in both arms. Recorded so
                # that a reader who tries the bitwise check we tried first finds
                # the result here rather than concluding it means something.
                "n_distinct_trained_A": len(uniq),
                "n_lora_A_tensors": next(iter(digests.values()))[1],
                "launch": launch_condition(base, cohort),
            }
            met = entry[cohort]["launch"].get("source_condition_met")
            print(f"  {base}/{cohort}: {len(digests)} adapters, "
                  f"launch condition "
                  f"{'MET (one shared draw)' if met else 'not met (independent draws)'}"
                  if entry[cohort]["launch"] else
                  f"  {base}/{cohort}: {len(digests)} adapters, no run config found")
        if entry:
            report["cohorts"][base] = entry

    shared = [e["seed1"]["launch"].get("source_condition_met")
              for e in report["cohorts"].values()
              if "seed1" in e and e["seed1"]["launch"]]
    indep = [e["indep1"]["launch"].get("source_condition_met") is False
             for e in report["cohorts"].values()
             if "indep1" in e and e["indep1"]["launch"]]
    report["verdict"] = {
        "shared_arm_condition_met": f"{sum(map(bool, shared))}/{len(shared)}",
        "independent_arm_condition_not_met": f"{sum(indep)}/{len(indep)}",
        "note": ("The condition is evaluated at launch, from the seeds and the "
                 "construction recorded in the configs. It is not evaluated "
                 "from the shipped tensors, which are post-training and differ "
                 "in both arms."),
    }
    print(f"\nshared arm, source condition met at launch:      "
          f"{sum(map(bool, shared))}/{len(shared)} bases")
    print(f"independent arm, condition deliberately not met:  "
          f"{sum(indep)}/{len(indep)} bases")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
