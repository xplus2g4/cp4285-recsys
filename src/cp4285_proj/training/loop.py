"""GPU-resident training loop with early stopping on calibration (spec §5.4).

This package never imports `evaluation.guard` and holds no `TestAccess` token, so it cannot read test.parquet.
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from cp4285_proj.data.labels import LABELS, LabelSpec
from cp4285_proj.metrics.accuracy import gauc_value, per_user_ndcg
from cp4285_proj.recommenders.base import (
    Batch,
    CandidateSet,
    FitContext,
    FitResult,
    ItemTensors,
    Recommender,
    SelectionFn,
    TorchRecommender,
    TrainData,
)

EXTRA_COLUMNS = ("time_ms", "tab", "is_hate")
SELECTION_NDCG_K = 10  # selection for training labels without a binary relevance (raw_watch_time)


def configure_determinism(deterministic: bool) -> None:
    """Must run before the first CUDA call: cuBLAS reads CUBLAS_WORKSPACE_CONFIG at context creation."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_data(frames: dict[str, pl.DataFrame], label: LabelSpec, device: torch.device) -> TrainData:
    """Train split (+ item metadata and side features) → device tensors, built once."""
    train = frames["train"]

    def t(col: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(train[col].to_numpy(), device=device).to(dtype)

    def codes(df: pl.DataFrame, key: str) -> torch.Tensor:
        return torch.tensor(df.sort(key).drop(key).to_numpy(), device=device).to(torch.int64)

    extras = {c: t(c, torch.float32 if c in LABELS and LABELS[c].kind == "graded" else torch.int64) for c in LABELS if c != label.name}
    extras.update({c: t(c, torch.int64) for c in EXTRA_COLUMNS})
    it = frames["items"].sort("item_idx")
    return TrainData(
        user=t("user_idx", torch.int64),
        item=t("item_idx", torch.int64),
        target=t(label.name, torch.float32),
        extras=extras,
        items=ItemTensors(**{c: torch.tensor(it[c].to_numpy(), device=device).to(torch.int64) for c in ("pop_exposure", "pop_bin", "dur_bin", "duration_ms")}),
        user_features=codes(frames["user_features"], "user_idx"),
        item_features=codes(frames["item_features"], "item_idx"),
        n_users=frames["users"].height,
        n_items=it.height,
    )


def candidate_set(part: pl.DataFrame) -> CandidateSet:
    return CandidateSet(
        row_id=part["row_id"].to_numpy(),
        user_idx=part["user_idx"].to_numpy(),
        item_idx=part["item_idx"].to_numpy(),
        labels={c: part[c].to_numpy() for c in LABELS},
    )


def make_selection(cal: CandidateSet, label: LabelSpec) -> SelectionFn:
    """Pipeline-owned selection on calibration: GAUC under the training label's binary relevance, or, for a
    label without one, mean NDCG@`SELECTION_NDCG_K` under the label with the panel's gain (users with
    all-equal gains excluded, as in the panel)."""
    users, pos = np.unique(cal.user_idx, return_inverse=True)
    values = cal.labels[label.name]
    if label.has_relevance:
        rel = label.binary(values)

        def selection(model: Recommender) -> float:
            return gauc_value(pos, model.score(cal.user_idx, cal.item_idx), rel, len(users))

        return selection

    minmax = label.ndcg_gain == "user_minmax"

    def selection_ndcg(model: Recommender) -> float:
        ndcg = per_user_ndcg(pos, model.score(cal.user_idx, cal.item_idx), values, len(users), SELECTION_NDCG_K, minmax)
        return float(np.nanmean(ndcg))

    return selection_ndcg


def run(model: TorchRecommender, train: TrainData, ctx: FitContext) -> FitResult:
    cfg = model.train_cfg
    opt = model.configure_optimizer()
    gen = torch.Generator(device=ctx.device)
    gen.manual_seed(ctx.seed)
    n = len(train.user)
    best_score, best_epoch, best_state, stale, epoch = -math.inf, -1, None, 0, -1
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        model.backbone.train()
        perm = torch.randperm(n, device=ctx.device, generator=gen)
        total = torch.zeros((), device=ctx.device)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            batch = Batch(idx=idx, user=train.user[idx], item=train.item[idx], target=train.target[idx], train=train)
            loss = model.batch_loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
        train_loss = (total / n).item()
        selection = ctx.selection(model)
        ctx.log.log_metrics(
            {"train_loss": train_loss, "cal_selection": selection, "epoch_time_s": time.perf_counter() - t0}, step=epoch
        )
        if selection > best_score:
            best_score, best_epoch, stale = selection, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.backbone.load_state_dict(best_state)
    return FitResult(best_epoch=best_epoch, best_selection_score=float(best_score), epochs_run=epoch + 1)


def load_split_frames(directory: Path) -> dict[str, pl.DataFrame]:
    """Everything training may read: train, calibration, items, users, side features (never test)."""
    from cp4285_proj.data import schema

    return {
        "train": schema.load_train(directory),
        "calibration": schema.load_calibration(directory),
        "items": schema.load_items(directory),
        "users": schema.load_users(directory),
        "user_features": schema.load_user_features(directory),
        "item_features": schema.load_item_features(directory),
    }
