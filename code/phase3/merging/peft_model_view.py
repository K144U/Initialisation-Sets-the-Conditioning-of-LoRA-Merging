"""Adapter from real PEFT PeftModel <-> FakePeftModel interface.

The merge_* functions in this package were written against FakePeftModel for
fast CPU testing. PeftModelView gives them the same 4 operations against a
real PeftModel:
  - layer_names(): list of LoRA-target module names
  - layer_topology[name] -> (out_dim, in_dim, rank)
  - get_delta(adapter, layer) -> full (out_dim x in_dim) tensor
  - add_adapter(name, factors): inject a new merged adapter into PEFT

Caveat: PEFT requires you to call add_weighted_adapter (or a similar setup
path) to allocate the merged adapter's LoraLayer module structure. We use
add_weighted_adapter as scaffolding (with arbitrary weights), then overwrite
the resulting A/B with our computed merged values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .tests._fake_model import FakeLoraLayer


@dataclass
class PeftModelView:
    """Adapts a real PEFT PeftModel to the merge_* function interface."""

    pmodel: object   # PeftModel instance (avoid circular import)

    def __post_init__(self) -> None:
        # peft import is local to keep CPU-only tests independent
        from peft.tuners.lora import LoraLayer
        self._LoraLayer = LoraLayer
        self._layers: Dict[str, object] = {}
        for name, mod in self.pmodel.named_modules():
            if isinstance(mod, LoraLayer):
                # use the parameter-qualified name, e.g. "...layers.0.self_attn.q_proj"
                self._layers[name] = mod

        self.layer_topology: Dict[str, tuple] = {}
        for name, mod in self._layers.items():
            # take any existing adapter's A/B to infer shape; PEFT stores per-adapter,
            # but rank/in/out are shared across adapters of a layer
            ad = next(iter(mod.lora_A.keys()))
            A = mod.lora_A[ad].weight  # (r, in_dim)
            B = mod.lora_B[ad].weight  # (out_dim, r)
            r, in_dim = A.shape
            out_dim, r2 = B.shape
            assert r == r2, f"r mismatch in layer {name}: A={A.shape}, B={B.shape}"
            self.layer_topology[name] = (out_dim, in_dim, r)

    def layer_names(self) -> List[str]:
        return list(self._layers.keys())

    def base_weight(self, layer: str) -> torch.Tensor:
        """The frozen base weight tensor under the LoRA layer (for the
        rd_encoder full-rank residual patch). PEFT >= 0.7 wraps it in
        .base_layer; older versions expose .weight on the module itself."""
        mod = self._layers[layer]
        base = getattr(mod, "base_layer", mod)
        return base.weight

    def get_delta(self, adapter: str, layer: str) -> torch.Tensor:
        """Full (out_dim x in_dim) materialized delta on the same device as PEFT stores it."""
        mod = self._layers[layer]
        A = mod.lora_A[adapter].weight       # (r, in_dim), bf16 on GPU
        B = mod.lora_B[adapter].weight       # (out_dim, r), bf16 on GPU
        scaling = float(mod.scaling[adapter])
        with torch.no_grad():
            # stay on device; bf16 matmul is fine here (low-magnitude updates)
            return scaling * (B.detach() @ A.detach())

    def add_adapter(self, name: str, factors: Dict[str, FakeLoraLayer]) -> None:
        """Inject `factors` as a new adapter named `name` into the PEFT model."""
        # Scaffold: ask PEFT to create the adapter slot via add_weighted_adapter
        # with arbitrary weights; we'll overwrite the A/B values afterwards.
        existing = list(self.pmodel.peft_config.keys())
        if name in existing:
            self.pmodel.delete_adapter(name)
        seed_adapters = existing[:2] if len(existing) >= 2 else existing
        seed_weights = [0.0] * len(seed_adapters)
        self.pmodel.base_model.add_weighted_adapter(
            adapters=seed_adapters,
            weights=seed_weights,
            adapter_name=name,
            combination_type="linear",
        )
        # Overwrite with our computed values
        for layer_name, mod in self._layers.items():
            f = factors[layer_name]
            # Match dtype/device of the existing PEFT tensor
            target_A = mod.lora_A[name].weight
            target_B = mod.lora_B[name].weight
            target_A.data.copy_(f.A.to(dtype=target_A.dtype, device=target_A.device))
            target_B.data.copy_(f.B.to(dtype=target_B.dtype, device=target_B.device))
            # PEFT's scaling for the new adapter is whatever its peft_config says
            # (alpha/r ratio); our factors fold scaling into B (scaling=1.0), so
            # we set the model's scaling to 1.0 to avoid double-scaling at forward.
            mod.scaling[name] = 1.0
