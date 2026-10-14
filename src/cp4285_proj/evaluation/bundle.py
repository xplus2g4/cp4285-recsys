"""EvalBundle (spec §7.2): everything a metric may read, plus user-level resampling."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from cp4285_proj.data.labels import LABELS, LabelSpec
from cp4285_proj.data.schema import GROUPINGS, CSRMask


@dataclass(frozen=True)
class ItemMeta:
    """Item metadata as numpy arrays indexed by item_idx."""

    duration_ms: np.ndarray
    pop_exposure: np.ndarray
    pop_bin: np.ndarray
    dur_bin: np.ndarray
    is_head: np.ndarray
    is_clickbait: np.ndarray

    @classmethod
    def from_frame(cls, items: pl.DataFrame) -> ItemMeta:
        return cls(**{f: items[f].to_numpy() for f in cls.__dataclass_fields__})

    @functools.cached_property
    def duration_order(self) -> np.ndarray:
        """argsort(duration_ms), shared by every resample."""
        return np.argsort(self.duration_ms, kind="stable")


def segment_index(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenated ranges [starts[j], starts[j] + counts[j]) as one flat index array."""
    total = int(counts.sum())
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + offsets


@dataclass(frozen=True)
class EvalBundle:
    split: Literal["calibration", "test"]
    users: np.ndarray  # user_idx [U]; may repeat under bootstrap
    cand_user_pos: np.ndarray  # [R] index into `users` for each candidate row (non-decreasing)
    cand_item: np.ndarray  # [R] item_idx
    cand_score: np.ndarray  # [R]
    cand_labels: dict[str, np.ndarray]  # label name → [R] values
    labels: dict[str, LabelSpec]  # kind, binary relevance fn
    topk: np.ndarray  # [U, K_max] item_idx
    seen: CSRMask  # per user: train items (for "un-interacted" in RSP)
    items: ItemMeta  # numpy arrays by item_idx
    user_groups: dict[str, np.ndarray]  # "activity" | "mainstream" | "dur_pref" → [U] group labels
    hist_dist: dict[str, np.ndarray]  # "pop" | "dur" → [U, 5]; NaN rows for no_history
    n_items: int

    # resampling bookkeeping (not part of the metric contract)
    _root: EvalBundle | None = field(default=None, repr=False, compare=False)
    _pos: np.ndarray | None = field(default=None, repr=False, compare=False)
    _cache: dict[Any, np.ndarray] = field(default_factory=dict, repr=False, compare=False)
    _row_start: np.ndarray = field(init=False, repr=False, compare=False)
    _row_count: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.cand_user_pos) and np.any(np.diff(self.cand_user_pos) < 0):
            raise ValueError("candidate rows must be sorted by cand_user_pos")
        counts = np.bincount(self.cand_user_pos, minlength=len(self.users))
        object.__setattr__(self, "_row_count", counts)
        object.__setattr__(self, "_row_start", np.cumsum(counts) - counts)

    @property
    def n_candidates(self) -> np.ndarray:
        """|C_u| per bundle user."""
        return self._row_count

    def resample(self, positions: np.ndarray) -> EvalBundle:
        """Bundle over users[positions] (with repeats), keeping their candidate rows and top-K rows."""
        root = self._root or self
        pos = self._pos[positions] if self._pos is not None else np.asarray(positions)
        rows = segment_index(root._row_start[pos], root._row_count[pos])
        return EvalBundle(
            split=root.split,
            users=root.users[pos],
            cand_user_pos=np.repeat(np.arange(len(pos)), root._row_count[pos]),
            cand_item=root.cand_item[rows],
            cand_score=root.cand_score[rows],
            cand_labels={k: v[rows] for k, v in root.cand_labels.items()},
            labels=root.labels,
            topk=root.topk[pos],
            seen=root.seen,
            items=root.items,
            user_groups={k: v[pos] for k, v in root.user_groups.items()},
            hist_dist={k: v[pos] for k, v in root.hist_dist.items()},
            n_items=root.n_items,
            _root=root,
            _pos=pos,
        )

    def user_cache(self, key: Any, fn: Callable[[EvalBundle], np.ndarray]) -> np.ndarray:
        """Per-user quantity ([U, ...]) computed once on the un-resampled bundle and indexed for resamples."""
        root = self._root or self
        key = ("user", key)
        if key not in root._cache:
            root._cache[key] = fn(root)
        value = root._cache[key]
        return value if self._pos is None else value[self._pos]

    def memo(self, key: Any, fn: Callable[[EvalBundle], Any]) -> Any:
        """Quantity computed once per bundle (e.g. item frequencies shared by several metrics)."""
        key = ("memo", key)
        if key not in self._cache:
            self._cache[key] = fn(self)
        return self._cache[key]


def build_bundle(
    split: Literal["calibration", "test"],
    candidates: pl.DataFrame,
    part: pl.DataFrame,
    topk: pl.DataFrame,
    items: pl.DataFrame,
    users: pl.DataFrame,
    seen: CSRMask,
) -> EvalBundle:
    """Join saved predictions with the part's labels and user/item metadata.

    `candidates`: row_id, user_idx, item_idx, score. `part`: the split part (labels by row_id).
    `topk`: user_idx, rank, item_idx, score (long format).
    """
    label_cols = list(LABELS)
    joined = candidates.join(part.select("row_id", *label_cols), on="row_id", how="inner").sort("user_idx", "row_id")
    if joined.height != candidates.height or candidates.height != part.height:
        raise ValueError(f"{split}: candidate predictions do not match the part's rows")
    bundle_users = np.unique(joined["user_idx"].to_numpy())
    cand_user = joined["user_idx"].to_numpy()
    k_max = int(topk["rank"].max()) + 1
    top = topk.sort("user_idx", "rank")
    top_users = top["user_idx"].unique(maintain_order=True).to_numpy()
    if not np.array_equal(top_users, bundle_users) or top.height != len(bundle_users) * k_max:
        raise ValueError(f"{split}: topk.parquet users do not match the candidate users")
    meta = users.sort("user_idx")
    if not np.array_equal(meta["user_idx"].to_numpy(), np.arange(meta.height)):
        raise ValueError("users.parquet must be dense and sorted by user_idx")
    sub = meta[bundle_users]

    def dist(col: str) -> np.ndarray:
        out = np.full((len(bundle_users), 5), np.nan)
        lists = sub[col].to_list()
        for j, v in enumerate(lists):
            if v is not None:
                out[j] = v
        return out

    return EvalBundle(
        split=split,
        users=bundle_users,
        cand_user_pos=np.searchsorted(bundle_users, cand_user),
        cand_item=joined["item_idx"].to_numpy(),
        cand_score=joined["score"].to_numpy(),
        cand_labels={c: joined[c].to_numpy() for c in label_cols},
        labels=dict(LABELS),
        topk=top["item_idx"].to_numpy().reshape(len(bundle_users), k_max),
        seen=seen,
        items=ItemMeta.from_frame(items.sort("item_idx")),
        user_groups={g: sub[c].to_numpy().astype(str) for g, c in GROUPINGS.items()},
        hist_dist={"pop": dist("hist_pop_dist"), "dur": dist("hist_dur_dist")},
        n_items=items.height,
    )

