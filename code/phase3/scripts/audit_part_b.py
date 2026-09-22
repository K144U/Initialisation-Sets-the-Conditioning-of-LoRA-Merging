#!/usr/bin/env python3
"""Part B of notes/prereg_prevalence_2026-08-15.md: published merging cohorts.

The registered rule: walk the merging papers cited in related work in citation
order and audit the cohorts released by the first two that release adapters
meeting the Part C inclusion criteria; if fewer than two do, report how many
were checked and that the precondition cannot be discharged, without
substituting another corpus.

The walk (2026-09-23), in citation order, with what each paper releases:
  ilharco2023task          full fine-tuned CLIP checkpoints        no
  matena2022fisher         no checkpoints                          no
  yadav2023ties            (IA)^3 checkpoints, not LoRA            no
  yu2024dare               none; uses public full SFT models       no
  della2024                none; uses public full SFT models       no
  stoica2025knots          LoRA r=16, HuggingFace (hoffman-lab)    YES
  tspa2025                 no release found                        no
  domerging2025            no release found                        no
  panariello2025core       code only; evaluates on KnOTS's cohorts no (reuses KnOTS)
  tara2025                 no release found                        no
  arm2026streaming         no release found                        no
  kim2025tvq               full-model task vectors                 no
  bit1merging2025          quantised task vectors, not LoRA        no
  ortizjimenez2023         full fine-tuned CLIP                    no
  jang2025taskvectorbases  code only, full task vectors            no
  atm2024alternating       no release found                        no
  gradients2025            no release found                        no
  cao2026collapse          no release found                        no
  jin2023regmean           full fine-tunes                         no
  lorm2024                 code only                               no
  regmeanpp2025            no release found                        no
  systematic2025merging    public full checkpoints via mergekit    no

One paper qualifies, so by the rule Part B is NOT discharged. The KnOTS
release is still audited, since it is exactly the kind of cohort Part B was
written for and a later paper (Core Space) evaluates on it. Its three cohorts
(Llama-3-8B NLI x6, ViT-B/32 x8, ViT-L/14 x8) are measured with the
registered audit's phase-2 code, unchanged; only the output directory
differs, so Part C's files are untouched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_cohorts as audit  # noqa: E402

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
OUT = ROOT / "results" / "phase3" / "prevalence_part_b"

KNOTS = {
    "meta-llama/Meta-Llama-3-8B": [f"hoffman-lab/KnOTS-Llama3_8B_lora_R16_{t}"
                                   for t in ("mnli", "snli", "sick", "qnli", "rte", "scitail")],
    "ViT-B-32": [f"hoffman-lab/KnOTS-ViT-B-32_lora_R16_{t}" for t in
                 ("stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45",
                  "sun397", "svhn")],
    "ViT-L-14": [f"hoffman-lab/KnOTS-ViT-L-14_lora_R16_{t}" for t in
                 ("stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45",
                  "sun397", "svhn")],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit.OUT, audit.CACHE = OUT, OUT / "adapters"
    cohorts, rejected = [], {}
    for base, repos in KNOTS.items():
        members = []
        for repo in repos:
            cfg = audit.get_json(audit.RESOLVE.format(repo=repo, fname="adapter_config.json"))
            if not cfg:
                rejected[repo] = "no adapter_config.json"
                continue
            tm = cfg.get("target_modules")
            members.append({"repo": repo, "rank": cfg.get("r"),
                            "target_modules": sorted(tm) if isinstance(tm, list) else [str(tm)],
                            "declared_base": cfg.get("base_model_name_or_path")})
        cohorts.append({"namespace": "hoffman-lab", "base_model": base,
                        "members": members})
    (OUT / "cohorts.json").write_text(json.dumps(
        {"source": "Part B, KnOTS release", "cohorts": cohorts,
         "rejected": rejected}, indent=2))
    print(json.dumps({c["base_model"]: [(m["repo"].split("/")[1], m["rank"],
                                         m["declared_base"]) for m in c["members"]]
                      for c in cohorts}, indent=1))
    return audit.phase2()


if __name__ == "__main__":
    raise SystemExit(main())
