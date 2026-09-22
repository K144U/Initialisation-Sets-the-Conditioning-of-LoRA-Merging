# Which fine-tuning pipelines hand every run the same LoRA A, at their defaults?

2026-09-23. Review item 7, second half. This is a reading of source code, not
a measurement, which is why `prereg_new_cohorts_2026-09-23.md` excludes it
from its registered predictions. The condition checked is Part A of
`prereg_prevalence_2026-08-15.md`: PEFT's `reset_lora_parameters` draws A
with `kaiming_uniform_` from the global torch RNG, so two runs receive the
identical A exactly when the global RNG is in the same state as each LoRA layer
is built (same seed set beforehand, same model, same target modules, rank,
device and dtype).

## Findings

| pipeline (version read) | seed set before LoRA A is drawn? | default | collapsed by default? |
|---|---|---|---|
| **Unsloth** 2026.5.2 (installed, `unsloth/models/llama.py`) | yes: `get_peft_model(..., random_state = 3407)`; `transformers_set_seed(random_state)` at line 2865; `_get_peft_model(model, lora_config)` at line 3195 | 3407 | **yes** |
| **LLaMA-Factory** (main, `src/llamafactory/hparams/parser.py`) | yes: `transformers.set_seed(training_args.seed)` inside `get_train_args`, before model load | 42 (HF `TrainingArguments.seed`, `training_args.py` line 1246 in transformers 5.5.0) | **yes** |
| **TRL** 0.24.0 `SFTTrainer` with `peft_config` | no: the PEFT model is built before `super().__init__`, which is where `Trainer` calls `set_seed(self.args.seed)` (`trainer.py` line 402) | n/a | depends on the calling script |
| **axolotl** (main, `src/axolotl/train.py`, `utils/config/__init__.py`) | no seeding found before `setup_model_and_tokenizer` | not found | not established |
| **PyTorch** 2.5.1, no seed set at all | `torch.initial_seed()` differs per fresh process (checked twice: 5883770033409691069, 18092558287914691822) | random | **no** |

## Initialisations that collapse regardless of seed

PiSSA and OLoRA (both in PEFT as `init_lora_weights="pissa"` / `"olora"`)
compute A from the base weight (top singular vectors; QR), so every task
fine-tuned from the same base with the same targets and rank starts from the
identical A, whatever the seed. Tested empirically in Part P of
`prereg_new_cohorts_2026-09-23.md`.

**LoRA-FA** (frozen A) is often described as collapsing exactly; that is true
only when the seed is shared. With independent seeds the frozen A factors are
independent draws and stay so, because they never train. The paper should say
this rather than list LoRA-FA as collapsed by construction.

## What this means for the paper

The two most-used single-GPU fine-tuning stacks (Unsloth, and LLaMA-Factory at
its default seed) produce the collapsed regime for any user who fine-tunes
several tasks on one base without changing the seed. That is a mechanism for
the 37.8% public prevalence, and it turns "shared initialisation" from a
seeding accident into a default. It is a claim about source code at the
versions named, not about any particular released cohort.
