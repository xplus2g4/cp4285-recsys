"""SNIPS (Swaminathan & Joachims, NeurIPS '15; Schnabel et al., ICML '16, Eq. 11) on the MF backbone (spec §13.4).

Same popularity-propensity weights and clipping as IPS (spec §13.3). The loss divides by the batch weight sum,
Σ w·ℓ / Σ w, instead of IPS's fixed normaliser. The propensities do not depend on the model, so over the full
train set the SNIPS normaliser is a constant and both estimators share a minimiser. They differ through the
per-batch self-normalisation: each batch's loss is a convex combination of its per-row losses, so a batch that
draws a few very large weights is not scaled up by them.
"""

from __future__ import annotations

import torch
from torch import nn

from cp4285_proj.recommenders.ips import IPS
from cp4285_proj.recommenders.registry import register_recommender


@register_recommender("snips")
class SNIPS(IPS):
    def loss(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        per_row = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (per_row * weight).sum() / weight.sum()
