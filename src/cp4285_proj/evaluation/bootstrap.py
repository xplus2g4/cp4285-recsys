"""User-level bootstrap and seed aggregation (spec §7.4)."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from cp4285_proj.evaluation.bundle import EvalBundle
from cp4285_proj.evaluation.panel import Panel, PanelResult, aggregate_per_user, scalarize, user_weights


def resample_positions(n_users: int, resamples: int, seed: int) -> np.ndarray:
    """[B, U] user positions; identical for every run evaluated on the same users with the same seed."""
    return np.random.default_rng(seed).integers(0, n_users, size=(resamples, n_users))


def bootstrap_samples(panel: Panel, b: EvalBundle, result: PanelResult, progress: bool = True) -> tuple[list[str], np.ndarray]:
    """[B, S] resampled scalars. Per-user metrics resample the saved per-user vectors; global and
    per-group metrics are recomputed on `b.resample(positions)`."""
    names = list(result.scalars)
    col = {n: j for j, n in enumerate(names)}
    positions = resample_positions(len(b.users), panel.bootstrap.resamples, panel.bootstrap.seed)
    out = np.full((len(positions), len(names)), np.nan)
    per_user = [(i, i.metric) for i in panel.instances if i.metric.kind == "per_user"]
    others = [(i, i.metric) for i in panel.instances if i.metric.kind != "per_user"]
    for r, pos in enumerate(tqdm(positions, desc=f"bootstrap {b.split}", disable=not progress, mininterval=5)):
        for inst, m in per_user:
            w = user_weights(m, b)
            out[r, col[inst.id]] = aggregate_per_user(result.per_user[inst.id][pos], None if w is None else w[pos])
        if others:
            rb = b.resample(pos)
            for inst, m in others:
                for name, v in scalarize(inst, m.fn(rb, **inst.kwargs)).items():
                    out[r, col[name]] = v
    return names, out


def percentile_ci(samples: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - ci) / 2 * 100
    with np.errstate(invalid="ignore"):
        return np.nanpercentile(samples, a, axis=0), np.nanpercentile(samples, 100 - a, axis=0)


def aggregate_seeds(names: list[str], points: np.ndarray, samples: np.ndarray, ci: float) -> dict[str, dict]:
    """`points` [n_runs, S] point estimates; `samples` [n_runs, B, S] bootstrap draws on shared user positions.

    mean ± std (ddof=1) over seeds, and a percentile CI of the seed-mean (per resample, the mean over runs).
    """
    lo, hi = percentile_ci(np.nanmean(samples, axis=0) if len(samples) else samples, ci)
    std = points.std(axis=0, ddof=1) if len(points) > 1 else np.full(points.shape[1], np.nan)
    return {
        n: {
            "mean": float(points[:, j].mean()),
            "std": float(std[j]),
            "ci_low": float(lo[j]),
            "ci_high": float(hi[j]),
            "per_seed": [float(v) for v in points[:, j]],
        }
        for j, n in enumerate(names)
    }
