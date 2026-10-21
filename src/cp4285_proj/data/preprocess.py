"""Raw KuaiRand-Pure CSVs → frozen split (spec §4). Deterministic; writes nothing unless every check passes."""

from __future__ import annotations

import dataclasses
import json
import math
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
from cp4285_proj.data.schema import NO_HISTORY, PART_SCHEMA, USERS_SCHEMA, file_sha256, validate_schema

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
    "activity_other_users": 1_035,
}
ACTIVITY_LEVELS = frozenset(
    {
        "high_active",
        "full_active",
        "middle_active",
        "low_active",
        "single_low_active",
        "2_14_day_new",
        "day_new",
        "30day_retention",
        "UNKNOWN",
    }
)
ACTIVITY_MAP = {"high_active": "high", "full_active": "full", "middle_active": "middle"}  # everything else → other
SPLITS = ("guarded",)
PARTS = ("train", "calibration", "test")
OUTPUT_FILES = ("train", "calibration", "test", "items", "users")


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
    frames: dict[str, pl.DataFrame]  # train, calibration, test, items, users
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


def _tertile_labels(values: np.ndarray, keys: np.ndarray, valid: np.ndarray, names: tuple[str, str, str]) -> np.ndarray:
    """Equal-size tertiles over `valid` rows ranked by (value, key); others get no_history."""
    out = np.full(len(values), NO_HISTORY, dtype=object)
    idx = np.flatnonzero(valid)
    order = np.lexsort((keys[idx], values[idx]))
    rank = np.empty(len(idx), dtype=np.int64)
    rank[order] = np.arange(len(idx))
    out[idx] = np.asarray(names, dtype=object)[rank * 3 // len(idx)]
    return out


def sparse_spearman(user: np.ndarray, item: np.ndarray, count: np.ndarray, reference: np.ndarray, n_users: int) -> np.ndarray:
    """Spearman ρ between each user's dense per-item count vector (zeros included) and `reference`.

    `user, item, count` list the non-zero entries (unique (user, item)). Ranks use average ties, so the
    zero block of a user shares rank (z + 1) / 2 and non-zeros are ranked above it.
    """
    n = len(reference)
    s = stats.rankdata(reference)  # average ties
    m = (n + 1) / 2
    var_s = float(np.sum(s * s) - n * m * m)
    s_total = float(s.sum())
    df = pl.DataFrame({"u": user, "i": item, "c": count}).with_columns(r=pl.col("c").rank("average").over("u"))
    nnz = np.bincount(user, minlength=n_users).astype(np.float64)
    z = n - nnz
    r0 = (z + 1) / 2
    r = df["r"].to_numpy() + z[user]  # non-zero ranks sit above the zero block
    s_nz = s[item]
    sum_rs_nz = np.bincount(user, weights=r * s_nz, minlength=n_users)
    sum_s_nz = np.bincount(user, weights=s_nz, minlength=n_users)
    sum_r2_nz = np.bincount(user, weights=r * r, minlength=n_users)
    sum_rs = r0 * (s_total - sum_s_nz) + sum_rs_nz
    var_r = z * r0 * r0 + sum_r2_nz - n * m * m
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sum_rs - n * m * m) / np.sqrt(var_r * var_s)


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
    users = raw.users.select("user_id", "user_active_degree").with_row_index("user_idx")
    users = users.with_columns(pl.col("user_idx").cast(pl.Int32), pl.col("user_id").cast(pl.Int32))
    n_users = users.height
    levels = set(users["user_active_degree"].unique().to_list())
    if levels != ACTIVITY_LEVELS:
        raise PreprocessError(f"user_active_degree levels changed: {sorted(levels)}")

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

    # 6. item metadata (train only; is_clickbait from calibration only)
    t_item = train["item_idx"].to_numpy()
    pop_exposure = np.bincount(t_item, minlength=n_items)
    pop_play = np.bincount(t_item[train["raw_watch_time"].to_numpy() > 0], minlength=n_items)
    duration = pool["duration_ms"].to_numpy()
    pop_rank = _rank(pop_exposure, n_items)
    dur_rank = _rank(duration, n_items)
    n_head = math.ceil(cfg.head_frac * n_items)
    is_head = pop_rank >= n_items - n_head

    cb = cfg.clickbait
    c_item = calibration["item_idx"].to_numpy()
    c_n = np.bincount(c_item, minlength=n_items)
    supported = c_n >= cb.min_support
    with np.errstate(invalid="ignore", divide="ignore"):
        c_click = np.bincount(c_item, weights=calibration["click"].to_numpy(), minlength=n_items) / c_n
        c_wr = np.bincount(c_item, weights=calibration["watch_ratio"].to_numpy().astype(np.float64), minlength=n_items) / c_n
        c_hate = np.bincount(c_item, weights=calibration["is_hate"].to_numpy(), minlength=n_items) / c_n
    thr_click = float(np.quantile(c_click[supported], cb.click_rate_q))
    thr_wr = float(np.quantile(c_wr[supported], cb.watch_ratio_q))
    thr_hate = float(np.quantile(c_hate[supported], cb.hate_rate_q))
    # A zero hate threshold would make the hate arm vacuous (every item has rate >= 0); require rate > 0 too.
    high_hate = (c_hate >= thr_hate) & (c_hate > 0)
    is_clickbait = supported & (c_click >= thr_click) & ((c_wr <= thr_wr) | high_hate)

    items = pool.with_columns(
        pop_exposure=pl.Series(pop_exposure, dtype=pl.Int32),
        pop_play=pl.Series(pop_play, dtype=pl.Int32),
        pop_bin=pl.Series(_equal_bins(pop_rank, cfg.n_bins), dtype=pl.Int8),
        dur_bin=pl.Series(_equal_bins(dur_rank, cfg.n_bins), dtype=pl.Int8),
        is_head=pl.Series(is_head),
        is_cold=pl.Series(pop_exposure == 0),
        is_clickbait=pl.Series(is_clickbait),
    )
    items = validate_schema(items, "items")

    # 7. user metadata (train only)
    t_user = train["user_idx"].to_numpy()
    n_train_rows = np.bincount(t_user, minlength=n_users)
    pairs = train.group_by("user_idx", "item_idx").len().sort("user_idx", "item_idx")
    p_user = pairs["user_idx"].to_numpy()
    n_train_items = np.bincount(p_user, minlength=n_users)
    has_history = n_train_items >= cfg.min_history
    user_idx = users["user_idx"].to_numpy()

    mainstream = sparse_spearman(
        p_user.astype(np.int64), pairs["item_idx"].to_numpy().astype(np.int64), pairs["len"].to_numpy(), pop_exposure, n_users
    )
    mainstream = np.where(has_history, mainstream, np.nan)
    dur_pref = (
        train.select("user_idx", pl.Series("d", duration[t_item]))
        .group_by("user_idx")
        .agg(pl.col("d").median())
    )
    dur_pref_ms = np.full(n_users, np.nan)
    dur_pref_ms[dur_pref["user_idx"].to_numpy()] = dur_pref["d"].to_numpy()
    dur_pref_ms = np.where(has_history, dur_pref_ms, np.nan)

    def hist_dist(bins: np.ndarray) -> list[list[float] | None]:
        counts = np.zeros((n_users, cfg.n_bins))
        np.add.at(counts, (t_user, bins[t_item].astype(np.int64)), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = (counts / counts.sum(1, keepdims=True)).astype(np.float32)
        return [dist[u].tolist() if has_history[u] else None for u in range(n_users)]

    activity = np.array([ACTIVITY_MAP.get(a, "other") for a in users["user_active_degree"].to_list()], dtype=object)
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
            "activity_group": activity.tolist(),
            "mainstream_score": mainstream,
            "mainstream_group": _tertile_labels(mainstream, user_idx, has_history, ("low", "mid", "high")).tolist(),
            "dur_pref_ms": dur_pref_ms,
            "dur_pref_group": _tertile_labels(dur_pref_ms, user_idx, has_history, ("short", "mid", "long")).tolist(),
            "hist_pop_dist": hist_dist(items["pop_bin"].to_numpy()),
            "hist_dur_dist": hist_dist(items["dur_bin"].to_numpy()),
        },
        schema=USERS_SCHEMA,
    )
    users_out = validate_schema(users_out, "users")

    # 8. manifest body
    frames = {"train": train, "calibration": calibration, "test": test, "items": items, "users": users_out}
    manifest = _manifest(raw, cfg, split, frames, std_drop, rnd_drop, dropped_ids, random, thr=(thr_click, thr_wr, thr_hate), supported=supported)
    _check_expected(manifest)
    return Built(frames=frames, manifest=manifest)


def _manifest(raw, cfg, split, frames, std_drop, rnd_drop, dropped_ids, random, thr, supported) -> dict[str, Any]:
    items, users = frames["items"], frames["users"]
    n_items = items.height
    pop = items["pop_exposure"].to_numpy()
    cold_item = items["is_cold"].to_numpy()
    no_hist = (users["n_train_items"] < cfg.min_history).to_numpy()
    no_rows = (users["n_train_rows"] == 0).to_numpy()
    groups = {"activity": "activity_group", "mainstream": "mainstream_group", "dur_pref": "dur_pref_group"}

    def group_counts(sub: pl.DataFrame) -> dict[str, dict[str, int]]:
        return {g: dict(sorted(sub[c].value_counts().iter_rows())) for g, c in groups.items()}

    rows, user_counts, cold, rates, per_group = {}, {}, {}, {}, {}
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
            "no_history_users": int(no_hist[part_users].sum()),
        }
        rates[part] = _positive_rates(df)
        per_group[part] = group_counts(users.filter(pl.col("user_idx").is_in(pl.Series(part_users).implode())))
    per_group["all"] = group_counts(users)
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
        "user_groups": per_group,
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
        "item_meta": {
            "head_items": int(items["is_head"].sum()),
            "head_exposure_share": float(pop[items["is_head"].to_numpy()].sum() / pop.sum()),
            "pop_bin_sizes": np.bincount(items["pop_bin"].to_numpy(), minlength=cfg.n_bins).tolist(),
            "dur_bin_sizes": np.bincount(items["dur_bin"].to_numpy(), minlength=cfg.n_bins).tolist(),
            "clickbait": {
                "source": "calibration",
                "supported_items": int(supported.sum()),
                "click_rate_threshold": thr[0],
                "watch_ratio_threshold": thr[1],
                "hate_rate_threshold": thr[2],
                "hate_rate_requires_positive": True,
                "clickbait_items": int(items["is_clickbait"].sum()),
            },
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
        "activity_other_users": m["user_groups"]["all"]["activity"].get("other", 0),
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
