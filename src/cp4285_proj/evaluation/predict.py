"""Saved predictions (spec §6): candidate scores and top-K lists per split part."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cp4285_proj.data.schema import CSRMask
from cp4285_proj.recommenders.base import Recommender

CANDIDATES = "candidates.parquet"
TOPK = "topk.parquet"


def pred_dir(run_dir: Path, part: str) -> Path:
    return run_dir / "pred" / part


def has_predictions(run_dir: Path, part: str) -> bool:
    d = pred_dir(run_dir, part)
    return (d / CANDIDATES).exists() and (d / TOPK).exists()


def write_candidates(model: Recommender, part_df: pl.DataFrame, out: Path) -> None:
    user, item = part_df["user_idx"].to_numpy(), part_df["item_idx"].to_numpy()
    pl.DataFrame(
        {"row_id": part_df["row_id"], "user_idx": part_df["user_idx"], "item_idx": part_df["item_idx"], "score": model.score(user, item)}
    ).write_parquet(out)


def write_topk(model: Recommender, users: np.ndarray, k: int, exclude: CSRMask | None, out: Path) -> None:
    items, scores = model.recommend(users, k, exclude)
    pl.DataFrame(
        {
            "user_idx": pl.Series(np.repeat(users, k), dtype=pl.Int32),
            "rank": pl.Series(np.tile(np.arange(k), len(users)), dtype=pl.Int32),
            "item_idx": pl.Series(items.ravel(), dtype=pl.Int32),
            "score": pl.Series(scores.ravel(), dtype=pl.Float32),
        }
    ).write_parquet(out)


def write_part_predictions(model: Recommender, run_dir: Path, part: str, part_df: pl.DataFrame, k: int, exclude: CSRMask | None) -> None:
    d = pred_dir(run_dir, part)
    d.mkdir(parents=True, exist_ok=True)
    write_candidates(model, part_df, d / CANDIDATES)
    write_topk(model, np.unique(part_df["user_idx"].to_numpy()), k, exclude, d / TOPK)


def topk_users(run_dir: Path, part: str) -> np.ndarray:
    return pl.read_parquet(pred_dir(run_dir, part) / TOPK, columns=["user_idx"])["user_idx"].unique().sort().to_numpy()


def topk_width(run_dir: Path, part: str) -> int:
    return int(pl.read_parquet(pred_dir(run_dir, part) / TOPK, columns=["rank"])["rank"].max()) + 1


def load_predictions(run_dir: Path, part: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = pred_dir(run_dir, part)
    return pl.read_parquet(d / CANDIDATES), pl.read_parquet(d / TOPK)
