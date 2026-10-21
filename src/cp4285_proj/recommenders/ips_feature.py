"""IPS with a feature-based propensity (Schnabel et al., ICML '16, §5) on the MF backbone (spec §13.6).

The propensity model is a factorization machine (Rendle, ICDM '10) over the user and item categorical side
features. It uses no ids, so it cannot reproduce item popularity directly. In `fit` it learns to tell logged
train exposures (u, i) from counterfactual ones (u, j), with j uniform over the pool: one negative per logged
row, redrawn every epoch. With 1:1 sampling its odds estimate the density ratio P_log(i | u) / P_uniform(i | u),
so a row's inverse propensity is exp(−logit). Because each negative shares the positive's user, user-only terms
cannot separate the classes. User features therefore act through their interactions with item features, and the
propensity is user-dependent, unlike the popularity propensity. Weights, clipping and the loss are IPS's
(spec §13.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from torch import nn

from cp4285_proj.recommenders.base import FitContext, TrainData
from cp4285_proj.recommenders.ips import IPS
from cp4285_proj.recommenders.registry import register_recommender

PROPENSITY_BATCH = 65536
AUC_SAMPLE = 200_000  # logged rows (and as many fresh uniform negatives) for the logged propensity AUC


@dataclass(frozen=True)
class IPSFeatureModelConfig:
    embedding_dim: int = 64
    clip: float = math.inf  # as in IPS: largest weight relative to the smallest; inf = no clipping
    propensity_dim: int = 8  # FM factor size
    propensity_epochs: int = 15  # the FM loss plateaus by then on guarded (1.2123 at 10, 1.2065 at 15)
    propensity_lr: float = 1e-2


class FeatureFM(nn.Module):
    """logit(x) = b + Σ_f w[x_f] + Σ_{f<g} <v[x_f], v[x_g]> over categorical fields (one code per field)."""

    def __init__(self, cardinalities: list[int], dim: int) -> None:
        super().__init__()
        self.register_buffer("offsets", torch.tensor(np.r_[0, np.cumsum(cardinalities)[:-1]], dtype=torch.int64))
        total = int(sum(cardinalities))
        self.w = nn.Parameter(torch.zeros(total))
        self.v = nn.Parameter(torch.randn(total, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:  # codes [B, F] → [B]
        idx = codes + self.offsets
        v = self.v[idx]
        pairwise = 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(-1)
        return self.b + self.w[idx].sum(1) + pairwise


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    ranks = stats.rankdata(np.r_[pos, neg])
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@register_recommender("ips_feature")
class IPSFeature(IPS):
    model_config_cls = IPSFeatureModelConfig

    def inverse_propensity(self, train: TrainData, ctx: FitContext) -> torch.Tensor:
        cfg, device = self.model_cfg, ctx.device
        cardinalities = (torch.cat([train.user_features.max(0).values, train.item_features.max(0).values]) + 1).tolist()

        def codes(user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
            return torch.cat([train.user_features[user], train.item_features[item]], dim=1)

        fm = FeatureFM(cardinalities, cfg.propensity_dim).to(device)
        opt = torch.optim.Adam(fm.parameters(), lr=cfg.propensity_lr)
        gen = torch.Generator(device=device)
        gen.manual_seed(ctx.seed)
        bce = nn.functional.binary_cross_entropy_with_logits
        n = len(train.user)
        for epoch in range(cfg.propensity_epochs):
            perm = torch.randperm(n, device=device, generator=gen)
            negatives = torch.randint(0, train.n_items, (n,), device=device, generator=gen)
            total = torch.zeros((), device=device)
            for s in range(0, n, PROPENSITY_BATCH):
                idx = perm[s : s + PROPENSITY_BATCH]
                user = train.user[idx]
                pos, neg = fm(codes(user, train.item[idx])), fm(codes(user, negatives[idx]))
                loss = bce(pos, torch.ones_like(pos)) + bce(neg, torch.zeros_like(neg))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.detach() * len(idx)
            ctx.log.log_metrics({"propensity_loss": (total / n).item()}, step=epoch)

        with torch.no_grad():
            logit = torch.cat([fm(codes(train.user[s : s + PROPENSITY_BATCH], train.item[s : s + PROPENSITY_BATCH])) for s in range(0, n, PROPENSITY_BATCH)])
            sample = torch.randperm(n, device=device, generator=gen)[:AUC_SAMPLE]
            fresh = torch.randint(0, train.n_items, (len(sample),), device=device, generator=gen)
            neg = fm(codes(train.user[sample], fresh))
        inverse = torch.exp(-logit.double())
        pop = train.items.pop_exposure.double()
        ctx.log.log_metrics({
            "propensity_auc": _auc(logit[sample].cpu().numpy(), neg.cpu().numpy()),
            "propensity_pop_spearman": float(stats.spearmanr(inverse.cpu().numpy(), (pop.max() / pop[train.item]).cpu().numpy()).statistic),
        })
        return inverse
