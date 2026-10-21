"""Parquet schemas (spec §4.3) and loaders. `test.parquet` is readable only via `load_test(token)` (spec §9)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from cp4285_proj.evaluation.guard import TestAccess

PART_SCHEMA: dict[str, pl.DataType] = {
    "row_id": pl.Int64(),
    "user_idx": pl.Int32(),
    "item_idx": pl.Int32(),
    "time_ms": pl.Int64(),
    "tab": pl.Int8(),
    "click": pl.Int8(),
    "long_view": pl.Int8(),
    "satisfaction": pl.Int8(),
    "watch_ratio": pl.Float32(),
    "raw_watch_time": pl.Int64(),
    "is_hate": pl.Int8(),
}

ITEMS_SCHEMA: dict[str, pl.DataType] = {
    "item_idx": pl.Int32(),
    "video_id": pl.Int32(),
    "duration_ms": pl.Int64(),
    "pop_exposure": pl.Int32(),
    "pop_play": pl.Int32(),
    "pop_bin": pl.Int8(),
    "dur_bin": pl.Int8(),
    "is_head": pl.Boolean(),
    "is_cold": pl.Boolean(),
    "is_clickbait": pl.Boolean(),
}

USERS_SCHEMA: dict[str, pl.DataType] = {
    "user_idx": pl.Int32(),
    "user_id": pl.Int32(),
    "role": pl.String(),
    "n_train_rows": pl.Int32(),
    "n_train_items": pl.Int32(),
    "activity_group": pl.String(),
    "mainstream_score": pl.Float32(),
    "mainstream_group": pl.String(),
    "dur_pref_ms": pl.Float32(),
    "dur_pref_group": pl.String(),
    "hist_pop_dist": pl.List(pl.Float32()),
    "hist_dur_dist": pl.List(pl.Float32()),
}

# Categorical side features (spec §4.2 step 9). Every column is an Int32 code: 0 = missing, 1..K = the sorted
# distinct raw values. Static metadata only; `video_features_statistic_pure.csv` is never read.
USER_FEATURES: tuple[str, ...] = (
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    *(f"onehot_feat{k}" for k in range(18)),
)
ITEM_FEATURES: tuple[str, ...] = ("video_type", "upload_type", "upload_dt", "music_type", "tag", "orientation", "duration_decile")
USER_FEATURES_SCHEMA: dict[str, pl.DataType] = {"user_idx": pl.Int32(), **{c: pl.Int32() for c in USER_FEATURES}}
ITEM_FEATURES_SCHEMA: dict[str, pl.DataType] = {"item_idx": pl.Int32(), **{c: pl.Int32() for c in ITEM_FEATURES}}

SCHEMAS = {
    "train": PART_SCHEMA,
    "calibration": PART_SCHEMA,
    "test": PART_SCHEMA,
    "items": ITEMS_SCHEMA,
    "users": USERS_SCHEMA,
    "user_features": USER_FEATURES_SCHEMA,
    "item_features": ITEM_FEATURES_SCHEMA,
}

NO_HISTORY = "no_history"
GROUPINGS = {"activity": "activity_group", "mainstream": "mainstream_group", "dur_pref": "dur_pref_group"}
GROUP_LEVELS = {
    "activity": ("high", "full", "middle", "other"),
    "mainstream": ("low", "mid", "high", NO_HISTORY),
    "dur_pref": ("short", "mid", "long", NO_HISTORY),
}


def split_dir(split: str, processed_dir: str | Path = "data/processed") -> Path:
    return Path(processed_dir) / split


def validate_schema(df: pl.DataFrame, name: str) -> pl.DataFrame:
    expected = SCHEMAS[name]
    got = dict(df.schema)
    if list(got) != list(expected) or any(got[c] != expected[c] for c in expected):
        raise ValueError(f"{name}: schema mismatch\n expected {expected}\n got      {got}")
    return df


def _read(directory: Path, name: str) -> pl.DataFrame:
    path = directory / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `kr-preprocess --split {directory.name}` first")
    return validate_schema(pl.read_parquet(path), name)


def load_train(directory: Path) -> pl.DataFrame:
    return _read(directory, "train")


def load_calibration(directory: Path) -> pl.DataFrame:
    return _read(directory, "calibration")


def load_test(directory: Path, token: TestAccess) -> pl.DataFrame:
    """The only reader of test.parquet. `token` comes from `evaluation.guard.authorize`."""
    from cp4285_proj.evaluation.guard import check_token

    check_token(token)
    return _read(directory, "test")


def load_items(directory: Path) -> pl.DataFrame:
    return _read(directory, "items")


def load_users(directory: Path) -> pl.DataFrame:
    return _read(directory, "users")


def load_user_features(directory: Path) -> pl.DataFrame:
    return _read(directory, "user_features")


def load_item_features(directory: Path) -> pl.DataFrame:
    return _read(directory, "item_features")


def load_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(directory: Path) -> str:
    return file_sha256(directory / "manifest.json")


@dataclass(frozen=True)
class CSRMask:
    """Per-user item sets over all `n_rows` users (row = user_idx). Indices sorted and unique within a row."""

    indptr: np.ndarray  # int64 [n_rows + 1]
    indices: np.ndarray  # int32 [nnz]

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    @classmethod
    def from_pairs(cls, rows: np.ndarray, cols: np.ndarray, n_rows: int) -> CSRMask:
        pairs = np.unique(np.stack([rows.astype(np.int64), cols.astype(np.int64)]), axis=1)
        counts = np.bincount(pairs[0], minlength=n_rows)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return cls(indptr=indptr, indices=pairs[1].astype(np.int32))

    def gather(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(position in `rows`, item) for every masked entry of the given rows."""
        starts = self.indptr[rows]
        counts = self.indptr[rows + 1] - starts
        pos = np.repeat(np.arange(len(rows)), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        return pos, self.indices[np.repeat(starts, counts) + offsets]


def load_seen_mask(directory: Path, n_users: int) -> CSRMask:
    """Train items per user (for top-K masking and RSP's "un-interacted")."""
    train = pl.read_parquet(directory / "train.parquet", columns=["user_idx", "item_idx"])
    return CSRMask.from_pairs(train["user_idx"].to_numpy(), train["item_idx"].to_numpy(), n_users)
