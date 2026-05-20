"""Minimal mock PeftModel for fast CPU-only merging tests.

Avoids the full PEFT + transformers stack. Holds a dict of per-adapter LoRA
factors (A, B, scaling) keyed by (adapter_name, layer_name). Provides the
same `add_weighted_adapter`-ish API surface that our merge functions call.

This mock is INTENTIONALLY minimal — it covers only what our merging methods
need. Real PEFT methods are validated against the real PEFT stack in the Day 3.7
GPU smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class FakeLoraLayer:
    """One layer's LoRA factors for one adapter."""
    A: torch.Tensor          # (r, in_dim)
    B: torch.Tensor          # (out_dim, r)
    scaling: float = 1.0


@dataclass
class FakePeftModel:
    """A bag of adapters keyed by name.

    adapters[name][layer_name] -> FakeLoraLayer

    layer_topology gives (out_dim, in_dim, rank) per layer; identical across adapters.
    """
    layer_topology: Dict[str, tuple] = field(default_factory=dict)
    adapters: Dict[str, Dict[str, FakeLoraLayer]] = field(default_factory=dict)

    def add_adapter(self, name: str, factors: Dict[str, FakeLoraLayer]) -> None:
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} already exists")
        self.adapters[name] = factors

    def delete_adapter(self, name: str) -> None:
        if name in self.adapters:
            del self.adapters[name]

    def get_delta(self, adapter: str, layer: str) -> torch.Tensor:
        """Full materialized delta for one (adapter, layer)."""
        f = self.adapters[adapter][layer]
        return f.scaling * (f.B @ f.A)

    def layer_names(self) -> List[str]:
        return list(self.layer_topology.keys())


def make_random_fake_model(
    layer_specs: Dict[str, tuple],   # name -> (out_dim, in_dim, rank)
    adapter_names: List[str],
    rng: torch.Generator | None = None,
    scale: float = 0.01,
) -> FakePeftModel:
    """Construct a FakePeftModel with random LoRA factors per adapter."""
    if rng is None:
        rng = torch.Generator().manual_seed(20260518)
    model = FakePeftModel(layer_topology=dict(layer_specs))
    for ad in adapter_names:
        layer_dict = {}
        for layer, (out_dim, in_dim, r) in layer_specs.items():
            A = scale * torch.randn(r, in_dim, generator=rng, dtype=torch.float32)
            B = scale * torch.randn(out_dim, r, generator=rng, dtype=torch.float32)
            layer_dict[layer] = FakeLoraLayer(A=A, B=B, scaling=1.0)
        model.add_adapter(ad, layer_dict)
    return model


def make_zero_fake_model(
    layer_specs: Dict[str, tuple],
    adapter_names: List[str],
) -> FakePeftModel:
    """All-zero LoRA factors — for the zero-idempotence test."""
    model = FakePeftModel(layer_topology=dict(layer_specs))
    for ad in adapter_names:
        layer_dict = {}
        for layer, (out_dim, in_dim, r) in layer_specs.items():
            A = torch.zeros(r, in_dim, dtype=torch.float32)
            B = torch.zeros(out_dim, r, dtype=torch.float32)
            layer_dict[layer] = FakeLoraLayer(A=A, B=B, scaling=1.0)
        model.add_adapter(ad, layer_dict)
    return model


def make_identical_fake_model(
    layer_specs: Dict[str, tuple],
    adapter_names: List[str],
    rng: torch.Generator | None = None,
    scale: float = 0.01,
) -> FakePeftModel:
    """All adapters share the SAME LoRA factors — for the identity-collapse test."""
    if rng is None:
        rng = torch.Generator().manual_seed(20260518)
    model = FakePeftModel(layer_topology=dict(layer_specs))
    shared = {}
    for layer, (out_dim, in_dim, r) in layer_specs.items():
        A = scale * torch.randn(r, in_dim, generator=rng, dtype=torch.float32)
        B = scale * torch.randn(out_dim, r, generator=rng, dtype=torch.float32)
        shared[layer] = (A, B)
    for ad in adapter_names:
        layer_dict = {
            layer: FakeLoraLayer(A=shared[layer][0].clone(), B=shared[layer][1].clone(), scaling=1.0)
            for layer in layer_specs
        }
        model.add_adapter(ad, layer_dict)
    return model
