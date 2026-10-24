"""Write the B3 classification of the 18 collapsed public cohorts (by hand, from
collapsed_cohort_metadata.json: repo names, model cards, tags, configs)."""
import json
import sys

C = [
    # namespace, route evidence, route, what the members are
    ("ceselder", "card: second adapter is 'DPO on top of Best v3'", "derived (card)", "versions"),
    ("vaghawan", "names: '-best' and '-last' of the same runs; Unsloth tag", "checkpoints (name)", "versions"),
    ("Satya-Sri", "names: 'stage1_adapter' and 'sft_adapter'; Unsloth tag", "stages (name)", "versions"),
    ("Tevatron", "names: 'OmniEmbed-v0.1' and 'OmniEmbed-v0.1-multivent', identical cards", "variant (name)", "versions"),
    ("Journey9ni", "names: '-lora' and '-lora-vstibench'", "variant (name)", "versions"),
    ("LASR-Callum", "card: 'same seed, same hyperparameters' for a pair; data-mixture ablations", "seed (card)", "versions"),
    ("NAMAA-Space", "Unsloth + TRL tags; OCR model versions 0.1 and 0.2.2.1", "library (tag)", "versions"),
    ("EhtishamAhmed", "Unsloth + TRL tags; two data splits of one banking-FAQ task", "library (tag)", "versions"),
    ("ewinregirgojr", "Unsloth tags on 3 of 4; tool-use v1-v3 and a stock-analyst adapter", "library (tag)", "versions"),
    ("McGill-NLP", "LLM2Vec supervised and unsupervised SimCSE, both after the same MNTP stage", "same project", "versions"),
    ("Jazhyc", "ten versions of one judge adapter (v2 to v8)", "same project", "versions"),
    ("berencarkci", "names: 'va-sft' and 'va-sft-v5'", "same project", "versions"),
    ("hallisky", "sixteen writing-style adapters (formality, voice, sarcasm, ...)", "same project", "tasks"),
    ("lorahub", "196 FLAN-task adapters released for composition", "same project", "tasks"),
    ("adamkarvonen", "22 adapters: taboo words, PersonaQA, LatentQA", "same project", "tasks"),
    ("DataAttributionEval", "three datasets (XSTest variants, counterfactual)", "same project", "tasks"),
    ("nvidia", "Aegis safety policies: defensive and permissive", "same project", "tasks"),
    ("agentic-moral-alignment", "three GRPO game environments, same setup", "same project", "tasks"),
]

out = {
    "note": "Professor B3. Descriptive reading of public metadata, not registered. "
            "Route: the most specific evidence of how the members came to share A. "
            "Members: 'tasks' if they are adapters for distinct tasks or objectives one "
            "might merge, 'versions' if versions, stages, checkpoints or data ablations "
            "of one model. No non-random initialisation and no frozen or shared-by-design "
            "A was found in any readable config.",
    "source": "collapsed_cohort_metadata.json (code/phase3/scripts/collapsed_cohort_metadata.py)",
    "cohorts": [{"namespace": n, "evidence": e, "route": r, "members": m} for n, e, r, m in C],
}
tally = {}
for _, _, r, m in C:
    tally.setdefault(r, 0)
    tally[r] += 1
out["route_counts"] = tally
out["members_counts"] = {k: sum(m == k for *_, m in C) for k in ("tasks", "versions")}
assert len(C) == 18
open(sys.argv[1], "w").write(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("route_counts", "members_counts")}, indent=1))
