"""D2Q (Zhan et al., KDD '22) on the MF backbone (spec §13.2).

Train rows are split into M equal-frequency duration groups. Within group k the label is the watch-time
quantile Φ̂_k(w) ∈ (0, 1], and one MF model h = σ(f(u,i)) is fitted to it (soft-target BCE, the pipeline's
convention for [0, 1] targets; it has the same minimiser, E[Φ̂_k(w) | u, i], as the paper's MSE). The item id
already identifies the duration group, so no duration feature is needed.

Inference ranks by the predicted watch time Φ̂_k⁻¹(h) of the item's group (`rank_by = "watch_time"`, the
published D2Q), or by the duration-normalised quantile h itself (`rank_by = "quantile"`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from cp4285_proj.recommenders.base import Backbone, Batch, FitContext, FitResult, TrainData
from cp4285_proj.recommenders.naive_mf import MFBackbone, NaiveMF
from cp4285_proj.recommenders.registry import register_recommender

N_LEVELS = 1001  # quantile levels 0, 0.001, …, 1 of the stored inverse CDFs


@dataclass(frozen=True)
class D2QModelConfig:
    embedding_dim: int = 64
    n_groups: int = 30  # M: equal-frequency duration groups over train rows (fewer if durations tie at a cut)
    rank_by: Literal["watch_time", "quantile"] = "watch_time"


@dataclass(frozen=True)
class D2QTables:
    row_quantile: np.ndarray  # float32 [N] Φ̂_k(w) of each train row
    item_group: np.ndarray  # int64 [n_items] duration group of every pool item
    inverse_cdf: np.ndarray  # float32 [n_groups, N_LEVELS] watch-time quantiles of each group; unused groups are 0
    n_used: int  # groups actually formed (≤ n_groups)


def d2q_tables(row_duration: np.ndarray, watch_time: np.ndarray, item_duration: np.ndarray, n_groups: int) -> D2QTables:
    """Duration groups from the train rows, the quantile label of each row, and each group's inverse CDF.

    Cut j is the sorted row duration at rank ⌈(j/M)(N−1)⌉ (`method="higher"`), an actual duration; a group is
    [cut_j, cut_{j+1}), so equal durations share a group and no group is empty. Items outside the train
    duration range fall into the first or last group.
    """
    levels = np.arange(1, n_groups) / n_groups
    cuts = np.unique(np.quantile(row_duration, levels, method="higher"))
    cuts = cuts[cuts > row_duration.min()]
    row_group = np.searchsorted(cuts, row_duration, side="right")
    row_quantile = np.empty(len(watch_time), dtype=np.float32)
    inverse_cdf = np.zeros((n_groups, N_LEVELS), dtype=np.float32)
    for k in range(len(cuts) + 1):
        rows = np.flatnonzero(row_group == k)
        w = np.sort(watch_time[rows])
        row_quantile[rows] = np.searchsorted(w, watch_time[rows], side="right") / len(w)
        inverse_cdf[k] = np.quantile(w, np.linspace(0.0, 1.0, N_LEVELS))
    return D2QTables(row_quantile, np.searchsorted(cuts, item_duration, side="right"), inverse_cdf, len(cuts) + 1)


class D2QBackbone(MFBackbone):
    """MF plus each item's duration group and the groups' inverse CDFs (buffers, saved in best.pt)."""

    def __init__(self, n_users: int, n_items: int, dim: int, n_groups: int) -> None:
        super().__init__(n_users, n_items, dim)
        self.register_buffer("item_group", torch.zeros(n_items, dtype=torch.int64))
        self.register_buffer("inverse_cdf", torch.zeros(n_groups, N_LEVELS))

    def predicted_watch_time(self, h: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """Φ̂_group⁻¹(h), linear between the stored levels; `group` broadcasts against `h`."""
        pos = h * (N_LEVELS - 1)
        lo = pos.floor().clamp(max=N_LEVELS - 2)
        frac = pos - lo
        flat = self.inverse_cdf.view(-1)
        idx = group * N_LEVELS + lo.long()
        return flat[idx] * (1 - frac) + flat[idx + 1] * frac


@register_recommender("d2q")
class D2Q(NaiveMF):
    model_config_cls = D2QModelConfig
    supported_labels = ("raw_watch_time",)
    _tables: D2QTables | None = None  # set by fit; `load` restores the buffers from best.pt instead

    def build_backbone(self, n_users: int, n_items: int) -> Backbone:
        backbone = D2QBackbone(n_users, n_items, self.model_cfg.embedding_dim, self.model_cfg.n_groups)
        if self._tables is not None:
            backbone.item_group.copy_(torch.as_tensor(self._tables.item_group))
            backbone.inverse_cdf.copy_(torch.as_tensor(self._tables.inverse_cdf))
        return backbone

    def fit(self, train: TrainData, ctx: FitContext) -> FitResult:
        duration = train.items.duration_ms.cpu().numpy()
        self._tables = d2q_tables(duration[train.item.cpu().numpy()], train.target.cpu().numpy(), duration, self.model_cfg.n_groups)
        self._quantile = torch.as_tensor(self._tables.row_quantile, device=ctx.device)
        ctx.log.log_metrics({"d2q_groups": float(self._tables.n_used)})
        return super().fit(train, ctx)

    def batch_loss(self, batch: Batch) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(self.backbone(batch.user, batch.item), self._quantile[batch.idx])

    def adjust_scores(self, logits: torch.Tensor, user: torch.Tensor, item: torch.Tensor | None) -> torch.Tensor:
        if self.model_cfg.rank_by == "quantile":
            return logits
        group = self.backbone.item_group if item is None else self.backbone.item_group[item]
        return self.backbone.predicted_watch_time(torch.sigmoid(logits), group)
