"""Budgeted search on calibration → final seed reruns (spec §5.5)."""

from __future__ import annotations

import dataclasses
import itertools
import json
import time
import uuid
from pathlib import Path
from typing import Any

from cp4285_proj.tracking import mlflow
import numpy as np
import torch

from cp4285_proj import tracking
from cp4285_proj.config import RunConfig
from cp4285_proj.training.runner import prepare, train_one

TUNING_SEED = 0


def expand_grid(search: tuple[tuple[str, tuple[Any, ...]], ...], budget: int, sweep_seed: int) -> list[dict[str, Any]]:
    """Full grid; if larger than `budget`, `budget` points sampled without replacement with `sweep_seed`."""
    keys = [k for k, _ in search]
    points = [dict(zip(keys, values)) for values in itertools.product(*(v for _, v in search))]
    if len(points) > budget:
        chosen = np.sort(np.random.default_rng(sweep_seed).choice(len(points), size=budget, replace=False))
        points = [points[i] for i in chosen]
    return points


def run_sweep(cfg: RunConfig, device: torch.device) -> dict[str, Any]:
    pipe = cfg.pipeline
    sweep_id = f"{cfg.experiment.recommender}-{cfg.experiment.train_label}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    points = expand_grid(cfg.recommender.search, pipe.tuning_budget, pipe.sweep_seed)
    data = prepare(cfg, device)
    tracking.setup(pipe.tracking_uri, pipe.experiment)
    tuning: list[dict[str, Any]] = []
    with mlflow.start_run(run_name=sweep_id, tags={"stage": "sweep", "sweep_id": sweep_id, "final": "false",
                                                   "recommender": cfg.experiment.recommender, "train_label": cfg.experiment.train_label}) as parent:
        mlflow.log_params({"n_points": len(points), "tuning_budget": pipe.tuning_budget, "sweep_seed": pipe.sweep_seed})
        for point in points:
            run_id, fit = train_one(dataclasses.replace(cfg, recommender=cfg.recommender.with_point(point)),
                                    TUNING_SEED, data, device, final=False, stage="tuning", sweep_id=sweep_id)
            tuning.append({"point": point, "run_id": run_id, **fit})
        best = max(tuning, key=lambda t: t["best_selection_score"])  # first wins on ties (grid order)
        final_cfg = dataclasses.replace(cfg, recommender=cfg.recommender.with_point(best["point"]))
        mlflow.log_params({f"best.{k}": v for k, v in best["point"].items()})
        mlflow.log_metric("best_tuning_selection", best["best_selection_score"])
        finals = []
        for seed in pipe.final_seeds:
            run_id, fit = train_one(final_cfg, seed, data, device, final=True, stage="final", sweep_id=sweep_id)
            finals.append({"seed": seed, "run_id": run_id, **fit})
        summary = {"sweep_id": sweep_id, "parent_run_id": parent.info.run_id, "best_point": best["point"],
                   "best_tuning": best, "tuning": tuning, "final": finals}
        out = Path(pipe.sweeps_dir) / sweep_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep.json").write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_artifact(str(out / "sweep.json"))
    return summary
