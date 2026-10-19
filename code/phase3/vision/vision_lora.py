#!/usr/bin/env python3
"""CLIP ViT-B/32 LoRA cohorts for notes/prereg_vision_2026-10-19.md.

  prep                      download the 8 datasets (train/test splits) and CLIP
  train --task T --cohort C one adapter (init/trainer seeds from the cohort name)
  eval  --cohort C          every registered merge on one cohort, dense, 8 tasks
  geometry                  median principal cosine per cohort (CPU)

Cohorts: shared1-3 (init = trainer seed k for every task) and indep1-3 (init =
trainer seed 100k + task index). Hyperparameters are the registration's.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT / "code" / "phase3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CLIP = "openai/clip-vit-base-patch32"
TASKS = {"SUN397": ("tanganke/sun397", "data"), "Cars": ("tanganke/stanford_cars", "data"),
         "RESISC45": ("tanganke/resisc45", "data"), "EuroSAT": ("tanganke/eurosat", "data"),
         "SVHN": ("ufldl-stanford/svhn", "cropped_digits"), "GTSRB": ("tanganke/gtsrb", "data"),
         "MNIST": ("ylecun/mnist", "mnist"), "DTD": ("tanganke/dtd", "data")}
NAMES = list(TASKS)
OUT = ROOT / "artifacts" / "lora_vision"
RES = ROOT / "results" / "phase3" / "vision"
CACHE = ROOT / "cache" / "vision"
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]
R, ALPHA, LR, WD, BATCH, STEPS, WARMUP = 16, 32, 2e-4, 0.1, 128, 1000, 100
N_TEST, N_GRAM = 2000, 256


def seeds(cohort: str, task: str) -> tuple[int, int]:
    k = int(cohort[-1])
    s = k if cohort.startswith("shared") else 100 * k + NAMES.index(task)
    return s, s


def load_split(task, split):
    from datasets import load_dataset
    repo, prefix = TASKS[task]
    return load_dataset(repo, data_files={split: f"{prefix}/{split}-*.parquet"}, split=split)


def test_subset(task):
    ds = load_split(task, "test")
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:N_TEST].tolist()
    return ds.select(sorted(idx))


def classnames(ds):
    return [n.replace("_", " ") for n in ds.features["label"].names]


def heads(device):
    """Zero-shot head per task: mean normalised text embedding over templates."""
    from transformers import CLIPModel, CLIPTokenizer
    from templates_wudi import get_templates
    p = CACHE / "heads.pt"
    if p.exists():
        return {k: v.to(device) for k, v in torch.load(p).items()}
    model = CLIPModel.from_pretrained(CLIP).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP)
    out = {}
    for task in NAMES:
        names = classnames(load_split(task, "test"))
        rows = []
        for c in names:
            texts = [t(c) for t in get_templates(task)]
            with torch.no_grad():
                e = model.get_text_features(**tok(texts, padding=True, return_tensors="pt").to(device))
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            rows.append(m / m.norm())
        out[task] = torch.stack(rows) * model.logit_scale.exp()
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in out.items()}, p)
    return out


def loader(ds, proc, bs, shuffle, gen=None, workers=3):
    def collate(rows):
        imgs = [r["image"].convert("RGB") for r in rows]
        x = proc(images=imgs, return_tensors="pt")["pixel_values"]
        return x, torch.tensor([r["label"] for r in rows])
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=bs * STEPS,
                                             generator=gen) if shuffle else None
    return torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, collate_fn=collate,
                                       num_workers=workers, persistent_workers=False)


def accuracy(vision, head, ds, proc, device):
    correct = n = 0
    vision.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader(ds, proc, 256, False):
            f = vision(pixel_values=x.to(device)).image_embeds.float()
            f = f / f.norm(dim=-1, keepdim=True)
            correct += int(((f @ head.T).argmax(-1).cpu() == y).sum())
            n += len(y)
    return correct / n


def cmd_prep(_a):
    from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
    CLIPModel.from_pretrained(CLIP); CLIPImageProcessor.from_pretrained(CLIP); CLIPTokenizer.from_pretrained(CLIP)
    for t in NAMES:
        for sp in ("train", "test"):
            ds = load_split(t, sp)
            print(t, sp, len(ds), len(classnames(ds)), flush=True)
    heads("cuda" if torch.cuda.is_available() else "cpu")
    print("PREP_DONE", flush=True)


def cmd_train(a):
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                              get_cosine_schedule_with_warmup)
    device = "cuda"
    init_seed, train_seed = seeds(a.cohort, a.task)
    out = OUT / a.task / a.cohort
    out.mkdir(parents=True, exist_ok=True)
    proc = CLIPImageProcessor.from_pretrained(CLIP)
    head = heads(device)[a.task]
    vision = CLIPVisionModelWithProjection.from_pretrained(CLIP).to(device)
    torch.manual_seed(init_seed)
    vision = get_peft_model(vision, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0,
                                               target_modules=TARGETS))
    save_file({n: p.detach().float().cpu().clone() for n, p in vision.named_parameters()
               if "lora_A" in n}, str(out / "adapter_A0.safetensors"))
    params = [p for p in vision.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sch = get_cosine_schedule_with_warmup(opt, WARMUP, STEPS)
    gen = torch.Generator().manual_seed(train_seed)
    torch.manual_seed(train_seed)
    vision.train()
    t0, losses = time.time(), []
    for step, (x, y) in enumerate(loader(load_split(a.task, "train"), proc, BATCH, True, gen)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = vision(pixel_values=x.to(device)).image_embeds.float()
        f = f / f.norm(dim=-1, keepdim=True)
        loss = torch.nn.functional.cross_entropy(f @ head.T, y.to(device))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 50 == 0:
            losses.append(float(loss)); print(f"step {step} loss {float(loss):.4f}", flush=True)
    vision.save_pretrained(str(out))
    acc = accuracy(vision, head, test_subset(a.task), proc, device)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"train__{a.task}__{a.cohort}.json").write_text(json.dumps({
        "task": a.task, "cohort": a.cohort, "init_seed": init_seed, "train_seed": train_seed,
        "own_test_acc": acc, "losses": losses, "seconds": time.time() - t0}, indent=1))
    print(f"TRAIN_DONE {a.task} {a.cohort} acc {acc:.4f}", flush=True)


class VisionFake:
    """FakePeftModel-compatible view of the 8 adapters, with get_layer (the
    hook adapter-only RegMean needs; without it, it falls back silently)."""
    def __init__(self, cohort):
        from safetensors.torch import load_file
        from merging.tests._fake_model import FakeLoraLayer
        self.adapters, self.layer_topology = {}, {}
        for t in NAMES:
            sd = load_file(str(OUT / t / cohort / "adapter_model.safetensors"))
            layers = {}
            for ka in sorted(k for k in sd if ".lora_A." in k):
                name = ka.split(".lora_A.")[0]
                A, B = sd[ka].float().cuda(), sd[ka.replace(".lora_A.", ".lora_B.")].float().cuda()
                layers[name] = FakeLoraLayer(A=A, B=B, scaling=ALPHA / R)
                self.layer_topology[name] = (B.shape[0], A.shape[1], A.shape[0])
            self.adapters[t] = layers

    def layer_names(self):
        return list(self.layer_topology)

    def get_layer(self, a, layer):
        return self.adapters[a][layer]

    def get_delta(self, a, layer):
        f = self.adapters[a][layer]
        return f.scaling * (f.B @ f.A)

    def add_adapter(self, name, factors):
        self.adapters[name] = factors

    def delete_adapter(self, name):
        self.adapters.pop(name, None)


METHODS = {
    "ta": ("task_arithmetic", {}, 0.3),
    "ties": ("ties", {"density": 0.2, "majority_sign_method": "total"}, None),
    "tsv": ("tsv", {"alpha": 1.0}, None),
    **{f"rm_{tag}": ("regmean", {"ridge_lambda": lam}, None) for tag, lam in
       [("1em6", 1e-6), ("1em3", 1e-3), ("1em2", 1e-2), ("3em2", 3e-2),
        ("1em1", 1e-1), ("3em1", 3e-1), ("1", 1.0)]},
    "rm_a09": ("regmean", {"alpha": 0.9, "ridge_lambda": 0.0}, None),
    "rma": ("regmean_act", {"alpha": 0.9, "ridge_lambda": 0.0}, None),
    "wudi_pub": ("wudi", {"mode": "adam", "iters": 300, "lr": 1e-5}, None),
    "wudi_cf": ("wudi", {"mode": "closed", "omega": 1e-6}, None),
}


def module_of(vision, layer):
    m = vision
    for part in layer.replace("base_model.model.", "").split("."):
        m = getattr(m, part)
    return m


def cmd_eval(a):
    from merging import _adapter_utils as au
    from merging.registry import REGISTRY
    from eval.activation_grams import gram_key
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    device = "cuda"
    proc = CLIPImageProcessor.from_pretrained(CLIP)
    H = heads(device)
    tests = {t: test_subset(t) for t in NAMES}
    vision = CLIPVisionModelWithProjection.from_pretrained(CLIP).to(device)
    fake = VisionFake(a.cohort)
    orig = {l: module_of(vision, l).weight.detach().clone() for l in fake.layer_names()}

    def patch(deltas):
        for l, W in orig.items():
            module_of(vision, l).weight.data.copy_(W + deltas.get(l, 0))

    grams = None
    if "rma" in METHODS:
        grams = {}
        for t in NAMES:
            patch({l: fake.get_delta(t, l) for l in fake.layer_names()})
            acc_g, hooks = {}, []
            for l in fake.layer_names():
                def hook(mod, inp, _out, key=gram_key(l)):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    acc_g[key] = acc_g.get(key, 0) + x.T @ x
                hooks.append(module_of(vision, l).register_forward_hook(hook))
            ds = load_split(t, "train")
            sub = ds.select(range(min(N_GRAM, len(ds))))
            with torch.no_grad():
                for x, _ in loader(sub, proc, 64, False):
                    vision(pixel_values=x.to(device))
            for h in hooks:
                h.remove()
            grams[t] = acc_g
    RES.mkdir(parents=True, exist_ok=True)
    for tag, (method, kw, wscale) in METHODS.items():
        kw = dict(kw)
        if method == "regmean_act":
            kw["grams"] = grams
        w = [wscale] * 8 if wscale else [1 / 8] * 8
        au.REALIZE["untruncated"] = True
        try:
            REGISTRY[method](fake, NAMES, w, "__merged__", **kw)
        finally:
            au.REALIZE["untruncated"] = False
        merged = {l: f.B @ f.A for l, f in fake.adapters["__merged__"].items()}
        fake.delete_adapter("__merged__")
        patch(merged)
        accs = {t: accuracy(vision, H[t], tests[t], proc, device) for t in NAMES}
        (RES / f"eval__{a.cohort}__{tag}.json").write_text(json.dumps(
            {"cohort": a.cohort, "method": tag, "accs": accs,
             "mean": sum(accs.values()) / 8, "worst": min(accs.values())}, indent=1))
        print(f"{a.cohort} {tag:9s} mean {sum(accs.values())/8:.4f} worst {min(accs.values()):.4f}", flush=True)
    patch({})
    zs = RES / "eval__zeroshot.json"
    if not zs.exists():
        accs = {t: accuracy(vision, H[t], tests[t], proc, device) for t in NAMES}
        zs.write_text(json.dumps({"accs": accs, "mean": sum(accs.values()) / 8}, indent=1))
    (RES / f"eval__{a.cohort}__DONE").write_text("ok")
    print("EVAL_DONE", a.cohort, flush=True)


def cmd_geometry(_a):
    from safetensors.torch import load_file
    out = {}
    for cohort in ["shared1", "shared2", "shared3", "indep1", "indep2", "indep3"]:
        mats = [{k: v.double() for k, v in load_file(str(OUT / t / cohort / "adapter_model.safetensors")).items()
                 if ".lora_A." in k} for t in NAMES]
        keys = sorted(set.intersection(*[set(m) for m in mats]))
        cos = []
        for k in keys:
            Q = [torch.linalg.qr(m[k].T)[0] for m in mats]
            for i in range(8):
                for j in range(i + 1, 8):
                    cos.append(float(torch.linalg.svdvals(Q[i].T @ Q[j]).clamp(max=1).median()))
        cos.sort()
        out[cohort] = cos[len(cos) // 2]
        print(cohort, out[cohort], flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "geometry.json").write_text(json.dumps(out, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval", "geometry"])
    ap.add_argument("--task")
    ap.add_argument("--cohort")
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval, "geometry": cmd_geometry}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
