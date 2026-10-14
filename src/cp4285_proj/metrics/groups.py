"""User-group metrics (spec §7.3): NDCG by group and Steck's calibration KL."""

from __future__ import annotations

from typing import Literal

import numpy as np

from cp4285_proj.data.schema import GROUP_LEVELS, NO_HISTORY
from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.metrics.accuracy import ndcg
from cp4285_proj.metrics.common import list_bin_counts
from cp4285_proj.metrics.registry import register_metric

CALIBRATION_ALPHA = 0.01


def group_means(values: np.ndarray, codes: np.ndarray, levels: tuple[str, ...]) -> dict[str, float]:
    """Mean of non-NaN values per group (`codes` index `levels`; NaN for an empty group), plus
    gap = max − min over groups other than no_history."""
    ok = ~np.isnan(values)
    sums = np.bincount(codes[ok], weights=values[ok], minlength=len(levels))
    counts = np.bincount(codes[ok], minlength=len(levels))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    out = {level: float(means[j]) for j, level in enumerate(levels)}
    real = [out[g] for g in levels if g != NO_HISTORY and not np.isnan(out[g])]
    out["gap"] = float(max(real) - min(real)) if real else float("nan")
    return out


def group_codes(b: EvalBundle, grouping: str) -> np.ndarray:
    """[U] index into GROUP_LEVELS[grouping] for each bundle user."""
    levels = GROUP_LEVELS[grouping]

    def encode(root: EvalBundle) -> np.ndarray:
        labels = root.user_groups[grouping]
        unknown = set(np.unique(labels)) - set(levels)
        if unknown:
            raise ValueError(f"unknown {grouping} groups {sorted(unknown)}")
        lookup = {level: j for j, level in enumerate(levels)}
        return np.array([lookup[g] for g in labels], dtype=np.int64)

    return b.user_cache(("group_codes", grouping), encode)


@register_metric(
    "ndcg_by_group",
    kind="per_group",
    params={"k": int, "grouping": Literal["activity", "mainstream", "dur_pref"]},
    needs_label=True,
)
def ndcg_by_group(b: EvalBundle, *, k: int, label: str, grouping: str) -> dict[str, float]:
    return group_means(ndcg(b, k=k, label=label), group_codes(b, grouping), GROUP_LEVELS[grouping])


def calibration_kl(p: np.ndarray, q: np.ndarray, alpha: float = CALIBRATION_ALPHA) -> np.ndarray:
    """Row-wise KL(p ‖ q̃), q̃ = (1−α)q + αp, with 0·log 0 = 0. NaN rows of p give NaN."""
    q_tilde = (1 - alpha) * q + alpha * p
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.where(p > 0, p * np.log(p / q_tilde), 0.0)
    out = terms.sum(1)
    out[np.isnan(p).any(1)] = np.nan
    return out


def _list_dist(b: EvalBundle, k: int, bins: np.ndarray) -> np.ndarray:
    return list_bin_counts(b, k, bins.astype(np.int64)) / k


@register_metric("calibration_kl_pop", kind="per_user", params={"k": int}, needs_label=False)
def calibration_kl_pop(b: EvalBundle, *, k: int) -> np.ndarray:
    return calibration_kl(b.hist_dist["pop"], _list_dist(b, k, b.items.pop_bin))


@register_metric("calibration_kl_dur", kind="per_user", params={"k": int}, needs_label=False)
def calibration_kl_dur(b: EvalBundle, *, k: int) -> np.ndarray:
    return calibration_kl(b.hist_dist["dur"], _list_dist(b, k, b.items.dur_bin))

