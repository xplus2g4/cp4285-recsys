"""IPS with a popularity propensity (Schnabel et al., ICML '16) on the MF backbone (spec §13.3).

Each logged exposure is weighted by its inverse exposure propensity, so the weighted training loss estimates the
loss under uniform item exposure, the condition of the random-exposure log. The popularity propensity of an item
is its share of train exposures relative to the most exposed item, p_i = n_i / max_j n_j.

Weights: w = 1/p, rescaled so that the smallest weight is 1, clipped at `clip` (Strehl et al. 2010; the
bias–variance trade-off of Schnabel et al. Thm 5.2), then divided by their train mean. Because the train mean is a
constant, the IPS loss mean(w·ℓ) keeps the unbiased estimator's normaliser fixed; SNIPS (spec §13.4) replaces it by
the batch Σw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from cp4285_proj.recommenders.base import Batch, FitContext, FitResult, TrainData
from cp4285_proj.recommenders.naive_mf import NaiveMF
from cp4285_proj.recommenders.registry import register_recommender


@dataclass(frozen=True)
class IPSModelConfig:
    embedding_dim: int = 64
    clip: float = math.inf  # largest weight relative to the smallest, before the mean normalisation; inf = no clipping


def ips_weights(inverse_propensity: torch.Tensor, clip: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Row weights (train mean 1) and their diagnostics: max weight, effective sample size share, clipped share."""
    w = inverse_propensity.double() / inverse_propensity.min()
    clipped = w > clip
    w = w.clamp(max=clip)
    w = w / w.mean()
    diagnostics = {
        "ips_max_weight": float(w.max()),
        "ips_ess_share": float(w.sum() ** 2 / (w**2).sum() / len(w)),
        "ips_clipped_share": float(clipped.double().mean()),
    }
    return w.float(), diagnostics


@register_recommender("ips")
class IPS(NaiveMF):
    model_config_cls = IPSModelConfig
    supported_labels = ("click", "long_view", "satisfaction", "watch_ratio")

    def inverse_propensity(self, train: TrainData, ctx: FitContext) -> torch.Tensor:
        """1/p per train row, up to a constant factor: here n_max / n_i of the row's item."""
        pop = train.items.pop_exposure.double()
        return pop.max() / pop[train.item]

    def fit(self, train: TrainData, ctx: FitContext) -> FitResult:
        self._weight, diagnostics = ips_weights(self.inverse_propensity(train, ctx), self.model_cfg.clip)
        ctx.log.log_metrics(diagnostics)
        return super().fit(train, ctx)

    def sample_weight(self, batch: Batch) -> torch.Tensor:
        return self._weight[batch.idx]

    def loss(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        return (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).mean()
