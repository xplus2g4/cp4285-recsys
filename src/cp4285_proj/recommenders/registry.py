"""Recommender registry: `@register_recommender(name)` and lookup by name."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

_REGISTRY: dict[str, type] = {}


def register_recommender(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"recommender {name!r} registered twice")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _ensure_builtins() -> None:
    from cp4285_proj.recommenders import d2q, ips, ips_feature, macr, naive_mf, pda, snips  # noqa: F401  (register every builtin)


def get_recommender(name: str) -> type:
    _ensure_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown recommender {name!r}; registered: {sorted(_REGISTRY)}") from None


def load_recommender(path: Path, device: torch.device):
    """Load a saved model, dispatching on the recommender name stored in the checkpoint."""
    name = torch.load(path, map_location="cpu", weights_only=True)["recommender"]
    return get_recommender(name).load(path, device)
