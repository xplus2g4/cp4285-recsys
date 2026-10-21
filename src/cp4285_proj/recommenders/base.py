"""Recommender abstraction (spec §5.2). The pipeline only uses the `Recommender` surface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

import numpy as np
import torch
from torch import nn

from cp4285_proj.data.schema import CSRMask

if TYPE_CHECKING:
    from cp4285_proj.config import TrainConfig
    from cp4285_proj.data.labels import LabelSpec

SCORE_CHUNK = 1 << 20  # (user, item) pairs per `score` forward
RECOMMEND_CHUNK = 2048  # users per `recommend` score_all


@dataclass(frozen=True)
class ItemTensors:
    """Train-derived item metadata on device, indexed by item_idx."""

    pop_exposure: torch.Tensor  # int64 [n_items]
    pop_bin: torch.Tensor  # int64 [n_items]
    dur_bin: torch.Tensor  # int64 [n_items]
    duration_ms: torch.Tensor  # int64 [n_items]


@dataclass(frozen=True)
class CandidateSet:
    """Logged (user, item) rows of a split part with their labels (numpy, row-aligned)."""

    row_id: np.ndarray  # int64 [R]
    user_idx: np.ndarray  # int32 [R]
    item_idx: np.ndarray  # int32 [R]
    labels: dict[str, np.ndarray]  # label name → [R]


class RunLogger(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...


SelectionFn = Callable[["Recommender"], float]


@dataclass(frozen=True)
class TrainData:
    """Train split, resident on `device`. Row-aligned tensors."""

    user: torch.Tensor  # int64 [N]
    item: torch.Tensor  # int64 [N]
    target: torch.Tensor  # float32 [N]; the training label (soft target for watch_ratio)
    extras: dict[str, torch.Tensor]  # other label columns, time, tab: for future weighting hooks
    items: ItemTensors  # pop_exposure, pop_bin, dur_bin, duration_ms on device, indexed by item_idx
    n_users: int
    n_items: int


@dataclass(frozen=True)
class Batch:
    """One index slice of `TrainData`; extras are gathered on demand."""

    idx: torch.Tensor  # int64 [B] row indices into TrainData
    user: torch.Tensor
    item: torch.Tensor
    target: torch.Tensor
    train: TrainData

    def extra(self, name: str) -> torch.Tensor:
        return self.train.extras[name][self.idx]


@dataclass(frozen=True)
class FitContext:
    label: LabelSpec
    device: torch.device
    seed: int
    selection: SelectionFn  # pipeline-owned: (Recommender) -> float, calibration GAUC
    calibration: CandidateSet  # for selection only; the model never sees calibration labels as training data
    log: RunLogger  # MLflow + file logging


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_selection_score: float
    epochs_run: int


class Recommender(Protocol):
    """The only surface the pipeline uses. Implementations register with @register_recommender(name)."""

    name: ClassVar[str]

    def fit(self, train: TrainData, ctx: FitContext) -> FitResult: ...
    def score(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Row-aligned float32 scores for (user, item) pairs. Used for candidate sets."""
        ...

    def recommend(self, user_idx: np.ndarray, k: int, exclude: CSRMask | None) -> tuple[np.ndarray, np.ndarray]:
        """Top-k over the whole pool per user: (item_idx int32 [U,k], score float32 [U,k]), sorted descending.
        Items in `exclude` get score -inf before top-k."""
        ...

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self: ...


class Backbone(nn.Module):
    """User/item → logit. Shared across methods so that only the correction varies."""

    n_users: int
    n_items: int

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # [B] logits
        raise NotImplementedError

    def score_all(self, user: torch.Tensor) -> torch.Tensor:  # [B, n_items] logits
        """Default: chunked forward over all items. Backbones override when cheaper (MF: one matmul)."""
        items = torch.arange(self.n_items, device=user.device)
        out = torch.empty(len(user), self.n_items, device=user.device)
        step = max(1, SCORE_CHUNK // self.n_items)
        for s in range(0, len(user), step):
            u = user[s : s + step]
            out[s : s + step] = self.forward(u.repeat_interleave(self.n_items), items.repeat(len(u))).view(len(u), -1)
        return out


class TorchRecommender(Recommender):
    """Generic torch implementation. Subclasses provide `build_backbone` and may override hooks.
    Naive MF uses only the defaults.

    Construction (not part of the pipeline surface): `cls(model_cfg, train_cfg)` where `model_cfg` is an
    instance of the subclass's `model_config_cls`. `supported_labels` is checked at config load.
    """

    backbone: Backbone
    model_config_cls: ClassVar[type]
    supported_labels: ClassVar[tuple[str, ...]]

    def __init__(self, model: Any, train: TrainConfig) -> None:
        self.model_cfg = model
        self.train_cfg = train
        self.device = torch.device("cpu")

    def build_backbone(self, n_users: int, n_items: int) -> Backbone:
        raise NotImplementedError

    def configure_optimizer(self) -> torch.optim.Optimizer:  # default: Adam(lr, weight_decay)
        if self.train_cfg.optimizer != "adam":
            raise ValueError(f"unsupported optimizer {self.train_cfg.optimizer!r}")
        return torch.optim.Adam(self.backbone.parameters(), lr=self.train_cfg.lr, weight_decay=self.train_cfg.weight_decay)

    # hooks (defaults are the naive behaviour)
    def batch_loss(self, batch: Batch) -> torch.Tensor:
        """Training loss of one batch. Override when the training-time score is not the backbone logit
        (PDA's popularity term, MACR's branches) or the target is transformed (D2Q)."""
        return self.loss(self.backbone(batch.user, batch.item), batch.target, self.sample_weight(batch))

    def sample_weight(self, batch: Batch) -> torch.Tensor | None:  # default None → unweighted
        return None

    def loss(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """BCE with logits; with weights, Σ w·ℓ / Σ w over the batch."""
        if weight is None:
            return nn.functional.binary_cross_entropy_with_logits(logits, target)
        per_row = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (per_row * weight).sum() / weight.sum()

    def adjust_scores(self, logits: torch.Tensor, user: torch.Tensor, item: torch.Tensor | None) -> torch.Tensor:
        """Inference-time correction of backbone logits; `item is None` means `logits` is [B, n_items]."""
        return logits

    # fit → training.loop.run(self, train, ctx); score/recommend → backbone + adjust_scores + masking
    def fit(self, train: TrainData, ctx: FitContext) -> FitResult:
        from cp4285_proj.training.loop import run

        self.device = ctx.device
        self.backbone = self.build_backbone(train.n_users, train.n_items).to(ctx.device)
        return run(self, train, ctx)

    @torch.inference_mode()
    def score(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        self.backbone.eval()
        out = np.empty(len(user_idx), dtype=np.float32)
        for s in range(0, len(user_idx), SCORE_CHUNK):
            u = torch.tensor(user_idx[s : s + SCORE_CHUNK], dtype=torch.int64, device=self.device)
            i = torch.tensor(item_idx[s : s + SCORE_CHUNK], dtype=torch.int64, device=self.device)
            out[s : s + SCORE_CHUNK] = self.adjust_scores(self.backbone(u, i), u, i).float().cpu().numpy()
        return out

    @torch.inference_mode()
    def recommend(self, user_idx: np.ndarray, k: int, exclude: CSRMask | None) -> tuple[np.ndarray, np.ndarray]:
        self.backbone.eval()
        items = np.empty((len(user_idx), k), dtype=np.int32)
        scores = np.empty((len(user_idx), k), dtype=np.float32)
        for s in range(0, len(user_idx), RECOMMEND_CHUNK):
            chunk = np.asarray(user_idx[s : s + RECOMMEND_CHUNK], dtype=np.int64)
            u = torch.as_tensor(chunk, device=self.device)
            logits = self.adjust_scores(self.backbone.score_all(u), u, None).float()
            if exclude is not None:
                pos, col = exclude.gather(chunk)
                logits[torch.as_tensor(pos, device=self.device), torch.as_tensor(col, dtype=torch.int64, device=self.device)] = -torch.inf
            top = torch.topk(logits, k, dim=1, sorted=True)
            items[s : s + len(chunk)] = top.indices.int().cpu().numpy()
            scores[s : s + len(chunk)] = top.values.cpu().numpy()
        return items, scores

    def save(self, path: Path) -> None:
        import dataclasses

        torch.save(
            {
                "recommender": self.name,
                "hparams": {"model": dataclasses.asdict(self.model_cfg), "train": dataclasses.asdict(self.train_cfg)},
                "n_users": self.backbone.n_users,
                "n_items": self.backbone.n_items,
                "state_dict": self.backbone.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        from cp4285_proj.config import TrainConfig, from_dict

        ckpt = torch.load(path, map_location=device, weights_only=True)
        if ckpt["recommender"] != cls.name:
            raise ValueError(f"{path} holds a {ckpt['recommender']!r} model, not {cls.name!r}")
        model = cls(
            from_dict(cls.model_config_cls, ckpt["hparams"]["model"], "model"),
            from_dict(TrainConfig, ckpt["hparams"]["train"], "train"),
        )
        model.device = device
        model.backbone = model.build_backbone(ckpt["n_users"], ckpt["n_items"]).to(device)
        model.backbone.load_state_dict(ckpt["state_dict"])
        model.backbone.eval()
        return model
