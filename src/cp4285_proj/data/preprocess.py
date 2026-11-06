"""Raw KuaiRand-Pure CSVs → frozen split (spec §4). Deterministic; writes nothing unless every check passes."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from cp4285_proj.config import DataConfig
from cp4285_proj.data.labels import LABELS
from cp4285_proj.data.schema import ITEM_FEATURES, PART_SCHEMA, USER_FEATURES, USERS_SCHEMA, file_sha256, validate_schema

# Measured row counts (spec §4.1). `video_features_statistic_pure.csv` is deliberately absent: it is
# averaged over both log windows, so it leaks the test window, and is never read.
RAW_FILES: dict[str, tuple[str, int]] = {
    "standard_early": ("log_standard_4_08_to_4_21_pure.csv", 1_141_112),
    "standard_late": ("log_standard_4_22_to_5_08_pure.csv", 295_497),
    "random": ("log_random_4_22_to_5_08_pure.csv", 1_186_059),
    "users": ("user_features_pure.csv", 27_285),
    "videos": ("video_features_basic_pure.csv", 7_583),
}

# Facts about this release (spec §4), checked at runtime before anything is written.
EXPECTED = {
    "dropped_items": 239,
    "dropped_train_rows": 24_076,
    "dropped_random_rows": 36_920,
    "pool_items": 7_344,
    "random_users_without_train_raw": 1_075,  # before the item drop; measured on the raw logs
}
SPLITS = ("guarded",)
PARTS = ("train", "calibration", "test")
OUTPUT_FILES = ("train", "calibration", "test", "items", "users", "user_features", "item_features")
DURATION_DECILES = 10  # catalogue duration bins of the item features (pool ranks, no log rows)


class PreprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawData:
    standard: pl.DataFrame  # log_standard_4_08_to_4_21 (the guarded training window)
    random: pl.DataFrame
    users: pl.DataFrame
    videos: pl.DataFrame


@dataclass(frozen=True)
class Built:
    frames: dict[str, pl.DataFrame]  # train, calibration, test, items, users, user_features, item_features
    manifest: dict[str, Any]  # everything except input/output hashes


# --------------------------------------------------------------------------- reading


def read_raw(raw_dir: Path) -> tuple[RawData, dict[str, Any]]:
    """Read and row-count-check the raw files. Fails if any file is missing or has the wrong row count."""
    missing = [name for name, _ in RAW_FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise PreprocessError(f"missing raw files in {raw_dir}: {missing}")
    frames: dict[str, pl.DataFrame] = {}
    inputs: dict[str, Any] = {}
    for key, (name, expected) in RAW_FILES.items():
        path = raw_dir / name
        if key == "standard_late":  # not used by `guarded`; only its row count is verified
            rows = pl.scan_csv(path).select(pl.len()).collect().item()
        else:
            frames[key] = pl.read_csv(path)
            rows = frames[key].height
        if rows != expected:
            raise PreprocessError(f"{name}: {rows} rows, expected {expected}")
        inputs[key] = {"file": name, "rows": rows, "sha256": file_sha256(path)}
    raw = RawData(standard=frames["standard_early"], random=frames["random"], users=frames["users"], videos=frames["videos"])
    return raw, inputs


# --------------------------------------------------------------------------- helpers


def _rank(primary: np.ndarray, n: int) -> np.ndarray:
    """0-based rank by (primary, item_idx); item_idx follows video_id, so ties break by video_id."""
    order = np.lexsort((np.arange(n), primary))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


def _equal_bins(rank: np.ndarray, n_bins: int) -> np.ndarray:
    return (rank * n_bins // len(rank)).astype(np.int8)


def categorical_codes(col: pl.Series) -> pl.Series:
    """Int32 codes: 0 = missing, 1..K = the distinct values sorted as strings (deterministic)."""
    s = col.cast(pl.String)
    levels = sorted(s.drop_nulls().unique().to_list())
    return s.replace_strict(levels, list(range(1, len(levels) + 1)), default=0, return_dtype=pl.Int32).alias(col.name)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    qs = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {f"p{int(round(q * 100))}": float(v) for q, v in zip([0.10, 0.25, 0.50, 0.75, 0.90], qs)}


def _positive_rates(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for name, spec in LABELS.items():
        values = df[name].to_numpy()
        entry = {"mean": float(values.mean())}
        if spec.has_relevance:
            entry["relevance_rate"] = float(spec.binary(values).mean())
        out[name] = entry
    return out


# --------------------------------------------------------------------------- building


def _log_rows(df: pl.DataFrame, pool: pl.DataFrame, users: pl.DataFrame, cap: float) -> pl.DataFrame:
    """Raw log → part schema. row_id = 0-based row index in the raw CSV (stable, traceable)."""
    out = (
        df.with_row_index("row_id")
        .join(pool.select("video_id", "item_idx"), on="video_id", how="inner")
        .join(users.select("user_id", "user_idx"), on="user_id", how="inner")
        .sort("row_id")
        .with_columns([spec.expr(cap).alias(name) for name, spec in LABELS.items()])
        .select([pl.col(c).cast(t) for c, t in PART_SCHEMA.items()])
    )
    return validate_schema(out, "train")


def build_split(raw: RawData, cfg: DataConfig, split: str = "guarded") -> Built:
    """Pure function: raw frames → output frames + manifest body. Raises PreprocessError on a failed check."""
    if split not in SPLITS:
        raise PreprocessError(f"unknown split {split!r}; implemented: {SPLITS}")

    # 1. drop items with no duration
    videos = raw.videos.select("video_id", "video_duration")
    dropped_ids = videos.filter(pl.col("video_duration").is_null())["video_id"]
    pool = (
        videos.filter(pl.col("video_duration").is_not_null())
        .sort("video_id")
        .with_row_index("item_idx")
        .select(
            pl.col("item_idx").cast(pl.Int32),
            pl.col("video_id").cast(pl.Int32),
            pl.col("video_duration").alias("duration_ms"),
        )
    )
    if (pool["duration_ms"] != pool["duration_ms"].round()).any() or (pool["duration_ms"] <= 0).any():
        raise PreprocessError("video_duration must be a positive integer number of ms for every pool item")
    pool = pool.with_columns(pl.col("duration_ms").cast(pl.Int64))
    pool_ids = pool["video_id"].cast(pl.Int64).implode()
    n_items = pool.height

    std_drop = raw.standard.filter(~pl.col("video_id").is_in(pool_ids))
    rnd_drop = raw.random.filter(~pl.col("video_id").is_in(pool_ids))
    dropped_zero_duration = bool((std_drop["duration_ms"] == 0).all() and (rnd_drop["duration_ms"] == 0).all())
    kept_positive_duration = bool(
        (raw.standard.filter(pl.col("video_id").is_in(pool_ids))["duration_ms"] > 0).all()
        and (raw.random.filter(pl.col("video_id").is_in(pool_ids))["duration_ms"] > 0).all()
    )
    if not dropped_zero_duration or not kept_positive_duration:
        raise PreprocessError("dropped items must be exactly those whose log rows have duration_ms == 0")

    # 2. index
    users = raw.users.select("user_id").with_row_index("user_idx")
    users = users.with_columns(pl.col("user_idx").cast(pl.Int32), pl.col("user_id").cast(pl.Int32))
    n_users = users.height

    # 3–4. labels + train
    cap = cfg.watch_ratio_cap
    train = _log_rows(raw.standard, pool, users, cap)
    random = _log_rows(raw.random, pool, users, cap)
    if train.height != raw.standard.height - std_drop.height or random.height != raw.random.height - rnd_drop.height:
        raise PreprocessError("log rows lost when joining users (unknown user_id in a log)")

    # 5. calibration / test by user
    rnd_users = np.sort(random["user_idx"].unique().to_numpy())
    perm = np.random.default_rng(cfg.split_seed).permutation(rnd_users)
    n_cal = int(round(cfg.calibration_frac * len(rnd_users)))
    cal_users = np.sort(perm[:n_cal])
    test_users = np.sort(perm[n_cal:])
    in_cal = pl.col("user_idx").is_in(pl.Series(cal_users).implode())
    calibration = random.filter(in_cal)
    test = random.filter(~in_cal)
    if calibration.height + test.height != random.height:
        raise PreprocessError("calibration/test partition lost rows")

    # 6. item metadata (train only)
    t_item = train["item_idx"].to_numpy()
    pop_exposure = np.bincount(t_item, minlength=n_items)
    pop_play = np.bincount(t_item[train["raw_watch_time"].to_numpy() > 0], minlength=n_items)
    dur_rank = _rank(pool["duration_ms"].to_numpy(), n_items)

    items = pool.with_columns(
        pop_exposure=pl.Series(pop_exposure, dtype=pl.Int32),
        pop_play=pl.Series(pop_play, dtype=pl.Int32),
        is_cold=pl.Series(pop_exposure == 0),
    )
    items = validate_schema(items, "items")

    # 7. user metadata (train only)
    t_user = train["user_idx"].to_numpy()
    n_train_rows = np.bincount(t_user, minlength=n_users)
    n_train_items = np.bincount(train.group_by("user_idx", "item_idx").len()["user_idx"].to_numpy(), minlength=n_users)
    user_idx = users["user_idx"].to_numpy()
    role = np.full(n_users, "train_only", dtype=object)
    role[cal_users] = "calibration"
    role[test_users] = "test"
    users_out = pl.DataFrame(
        {
            "user_idx": user_idx,
            "user_id": users["user_id"],
            "role": role.tolist(),
            "n_train_rows": n_train_rows,
            "n_train_items": n_train_items,
        },
        schema=USERS_SCHEMA,
    )
    users_out = validate_schema(users_out, "users")

    # 8. categorical side features (static metadata; no log rows involved)
    user_features = pl.DataFrame({"user_idx": user_idx}, schema={"user_idx": pl.Int32}).with_columns(
        categorical_codes(raw.users[c]) for c in USER_FEATURES
    )
    video = pool.select("item_idx", "video_id").join(raw.videos.with_columns(pl.col("video_id").cast(pl.Int32)), on="video_id", how="left")
    orientation = (
        pl.when(pl.col("server_height") > pl.col("server_width"))
        .then(pl.lit("portrait"))
        .when(pl.col("server_height") < pl.col("server_width"))
        .then(pl.lit("landscape"))
        .otherwise(pl.lit("square"))
    )
    video = video.with_columns(orientation.alias("orientation"))
    item_features = pl.DataFrame({"item_idx": pool["item_idx"]}).with_columns(
        *(categorical_codes(video[c]) for c in ITEM_FEATURES if c != "duration_decile"),
        pl.Series("duration_decile", _equal_bins(dur_rank, DURATION_DECILES) + 1, dtype=pl.Int32),
    )
    user_features = validate_schema(user_features, "user_features")
    item_features = validate_schema(item_features, "item_features")

    # 9. manifest body
    frames = {
        "train": train,
        "calibration": calibration,
        "test": test,
        "items": items,
        "users": users_out,
        "user_features": user_features,
        "item_features": item_features,
    }
    manifest = _manifest(raw, cfg, split, frames, std_drop, rnd_drop, dropped_ids, random)
    _check_expected(manifest)
    return Built(frames=frames, manifest=manifest)


def _manifest(raw, cfg, split, frames, std_drop, rnd_drop, dropped_ids, random) -> dict[str, Any]:
    items, users = frames["items"], frames["users"]
    n_items = items.height
    cold_item = items["is_cold"].to_numpy()
    no_rows = (users["n_train_rows"] == 0).to_numpy()

    rows, user_counts, cold, rates = {}, {}, {}, {}
    for part in PARTS:
        df = frames[part]
        part_users = np.unique(df["user_idx"].to_numpy())
        part_items = df["item_idx"].to_numpy()
        rows[part] = df.height
        user_counts[part] = int(len(part_users))
        cold[part] = {
            "cold_items": int(np.unique(part_items[cold_item[part_items]]).size),
            "cold_item_rows": int(cold_item[part_items].sum()),
            "users_without_train_rows": int(no_rows[part_users].sum()),
        }
        rates[part] = _positive_rates(df)
    user_counts["train_only_role"] = int((users["role"] == "train_only").sum())
    user_counts["random_log"] = user_counts["calibration"] + user_counts["test"]
    user_counts["all"] = users.height

    rnd_freq = np.bincount(random["item_idx"].to_numpy(), minlength=n_items).astype(np.float64)
    expected = rnd_freq.sum() / n_items
    chi2 = float(((rnd_freq - expected) ** 2 / expected).sum())
    train_pairs = frames["train"].select("user_idx", "item_idx").unique()
    rnd_pairs = random.select("user_idx", "item_idx").unique()
    random_users_without_train = int(
        np.isin(np.unique(random["user_idx"].to_numpy()), np.flatnonzero(no_rows)).sum()
    )
    raw_without_train = int(
        raw.random.select("user_id").unique().join(raw.standard.select("user_id").unique(), on="user_id", how="anti").height
    )
    duration = items["duration_ms"].to_numpy()
    return {
        "split": split,
        "split_seed": cfg.split_seed,
        "config": dataclasses.asdict(cfg),
        "drops": {
            "items": int(dropped_ids.len()),
            "dropped_video_ids": sorted(int(v) for v in dropped_ids.to_list()),
            "train_rows": std_drop.height,
            "random_rows": rnd_drop.height,
        },
        "pool": {"items": n_items, "users": users.height},
        "rows": rows,
        "users": user_counts,
        "cold": {
            **cold,
            "pool_cold_items": int(cold_item.sum()),
            "random_users_without_train_rows": random_users_without_train,
            "random_users_without_train_rows_raw": raw_without_train,
        },
        "positive_rates": rates,
        "train_repeats": {
            "repeat_exposures": frames["train"].height - train_pairs.height,
            "random_pairs_also_in_train": rnd_pairs.join(train_pairs, on=["user_idx", "item_idx"]).height,
        },
        "checks": {
            "random_item_uniformity": {
                "n_items": n_items,
                "items_with_random_rows": int((rnd_freq > 0).sum()),
                "mean": float(rnd_freq.mean()),
                "cv": float(rnd_freq.std() / rnd_freq.mean()),
                "chi2": chi2,
                "dof": n_items - 1,
                "chi2_p_value": float(stats.chi2.sf(chi2, n_items - 1)),
            },
            "duration_quantiles_ms": {
                "train": _quantiles(duration[frames["train"]["item_idx"].to_numpy()]),
                "random": _quantiles(duration[random["item_idx"].to_numpy()]),
                "catalogue": _quantiles(duration),
            },
        },
    }


def _check_expected(m: dict[str, Any]) -> None:
    got = {
        "dropped_items": m["drops"]["items"],
        "dropped_train_rows": m["drops"]["train_rows"],
        "dropped_random_rows": m["drops"]["random_rows"],
        "pool_items": m["pool"]["items"],
        "random_users_without_train_raw": m["cold"]["random_users_without_train_rows_raw"],
    }
    bad = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
    if bad:
        raise PreprocessError(f"release facts differ from spec §4 (got, expected): {bad}")


# --------------------------------------------------------------------------- writing


def write_split(built: Built, inputs: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Write to a temp dir, hash, then atomically replace `out_dir`."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.parent / f".{out_dir.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    outputs = {}
    for name in OUTPUT_FILES:
        path = tmp / f"{name}.parquet"
        built.frames[name].write_parquet(path, statistics=False)
        outputs[name] = file_sha256(path)
    manifest = {"input_files": inputs, **built.manifest, "output_files": outputs}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return manifest


def run(cfg: DataConfig, split: str = "guarded", out_dir: Path | None = None) -> dict[str, Any]:
    raw, inputs = read_raw(Path(cfg.raw_dir))
    built = build_split(raw, cfg, split)
    return write_split(built, inputs, out_dir or Path(cfg.processed_dir) / split)
