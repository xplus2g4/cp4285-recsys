"""PD / PDA (Zhang et al., SIGIR '21) on the MF backbone (spec §13.1).

Training fits logit(u,i,t) = f(u,i) + γ·log m_i^t, where f is the MF backbone and m_i^t is item i's local
popularity in the time stage t of the row. Inference intervenes on popularity: rank by f(u,i) + γ̃·log m̃_i,
with m̃_i the last-stage popularity. γ̃ = 0 is PD (deconfounded ranking), γ̃ > 0 is PDA.

The paper's P(c|u,i,m) = ELU'(f)·m^γ with BPR is moved to logit space so that it works with the pipeline's
pointwise BCE on logged exposures: the odds factorise as e^f · m^γ. PD ranking by f stays exact, because
Σ_z P(z)·σ(f + γ·log z) is increasing in f.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cp4285_proj.recommenders.base import Backbone, Batch, FitContext, FitResult, TrainData
from cp4285_proj.recommenders.naive_mf import MFBackbone, NaiveMF
from cp4285_proj.recommenders.registry import register_recommender


@dataclass(frozen=True)
class PDAModelConfig:
    embedding_dim: int = 64
    n_stages: int = 10  # equal-time stages over the train window (the paper's 10)
    gamma: float = 0.1  # training exponent on local popularity
    gamma_infer: float = 0.1  # inference exponent on last-stage popularity; 0 = PD


def local_log_popularity(time_ms: np.ndarray, item: np.ndarray, n_items: int, n_stages: int) -> tuple[np.ndarray, np.ndarray]:
    """log m_i^t per train row (its own stage) and per item for the last stage.

    m_i^t is the item's share of the stage's exposures, add-one smoothed and scaled by `n_items` so that a
    uniformly exposed item has m = 1: m_i^t = (c_i^t + 1) / (N_t + n_items) · n_items.
    """
    t0 = time_ms.min()
    stage = (time_ms - t0) * n_stages // (time_ms.max() - t0 + 1)
    counts = np.zeros((n_stages, n_items), dtype=np.float64)
    np.add.at(counts, (stage, item), 1.0)
    log_m = np.log((counts + 1.0) / (counts.sum(1, keepdims=True) + n_items) * n_items)
    return log_m[stage, item].astype(np.float32), log_m[-1].astype(np.float32)


class PDABackbone(MFBackbone):
    """MF plus the last-stage log popularity used at inference (a buffer, so it is saved in best.pt)."""

    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__(n_users, n_items, dim)
        self.register_buffer("log_pop_infer", torch.zeros(n_items))


@register_recommender("pda")
class PDA(NaiveMF):
    model_config_cls = PDAModelConfig
    supported_labels = ("click", "long_view", "satisfaction", "watch_ratio")
    _last_log_pop: torch.Tensor | None = None  # set by fit; `load` restores the buffer from best.pt instead

    def build_backbone(self, n_users: int, n_items: int) -> Backbone:
        backbone = PDABackbone(n_users, n_items, self.model_cfg.embedding_dim)
        if self._last_log_pop is not None:
            backbone.log_pop_infer.copy_(self._last_log_pop)
        return backbone

    def fit(self, train: TrainData, ctx: FitContext) -> FitResult:
        row, last = local_log_popularity(
            train.extras["time_ms"].cpu().numpy(), train.item.cpu().numpy(), train.n_items, self.model_cfg.n_stages
        )
        self._row_log_pop = torch.as_tensor(row, device=ctx.device)
        self._last_log_pop = torch.as_tensor(last)
        return super().fit(train, ctx)

    def batch_loss(self, batch: Batch) -> torch.Tensor:
        logits = self.backbone(batch.user, batch.item) + self.model_cfg.gamma * self._row_log_pop[batch.idx]
        return self.loss(logits, batch.target, self.sample_weight(batch))

    def adjust_scores(self, logits: torch.Tensor, user: torch.Tensor, item: torch.Tensor | None) -> torch.Tensor:
        if self.model_cfg.gamma_infer == 0:
            return logits
        pop = self.backbone.log_pop_infer
        return logits + self.model_cfg.gamma_infer * (pop if item is None else pop[item])
